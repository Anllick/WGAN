import os, time, pickle, random, time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
import tensorlayer as tl
from model import *
from utils import *
from config import config, log_config

os.environ['CUDA_VISIBLE_DEVICES']='0,1,2'

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
logdir = config.VALID.logdir

ni = int(np.sqrt(batch_size))
Epoch1=[]
Epoch2=[]

def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        # print(b_imgs.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs

def train():
    ## create folders to save result images and trained model
    save_dir_ginit = "samples/{}_ginit".format(tl.global_flag['mode'])
    save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA ===========================###
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.tif', printable=False))
    train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.tif', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.tif', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.tif', printable=False))


    train_hr_imgs = read_all_imgs(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    train_lr_imgs = read_all_imgs(train_lr_img_list, path=config.TRAIN.lr_img_path, n_threads=32)

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [batch_size, 64, 64, 3], name='t_image_input_to_SRGAN_generator')
    t_target_image = tf.placeholder('float32', [batch_size, 256, 256, 3], name='t_target_image')

    net_g = SRGAN_g(t_image, is_train=True, reuse=False)
    net_d, logits_real = SRGAN_d(t_target_image, is_train=True, reuse=False)
    _,     logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=True)

    net_g.print_params(False)
    net_d.print_params(False)

    ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    t_target_image_224 = tf.image.resize_images(t_target_image, size=[224, 224], method=0, align_corners=False) # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
    t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False) # resize_generate_image_for_vgg

    net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224+1)/2, reuse=False)
    _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224+1)/2, reuse=True)

    ## test inference
    net_g_test = SRGAN_g(t_image, is_train=False, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================###

    # Wasserstein GAN Loss
    with tf.name_scope('w_loss/WARS_1'):
        d_loss =  tf.reduce_mean(logits_fake) - tf.reduce_mean(logits_real)
        tf.summary.scalar('w_loss', d_loss)

    merged = tf.summary.merge_all()

    g_gan_loss = -1e-3 * tf.reduce_mean(logits_fake)
    mse_loss = tl.cost.mean_squared_error(net_g.outputs , t_target_image, is_mean=True)
    vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)

    g_loss = mse_loss + vgg_loss + g_gan_loss

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    ## Pretrain
    # g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)
    g_optim_init = tf.train.RMSPropOptimizer(lr_v).minimize(mse_loss, var_list=g_vars)

    ## SRGAN
    # g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    # d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)

    g_optim = tf.train.RMSPropOptimizer(lr_v).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.RMSPropOptimizer(lr_v).minimize(d_loss, var_list=d_vars)

    # clip op
    clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]


    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    loss_writer = tf.summary.FileWriter(logdir, sess.graph)

    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_{}.npz'.format(tl.global_flag['mode']), network=net_g) is False:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_{}_init.npz'.format(tl.global_flag['mode']), network=net_g)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/d_{}.npz'.format(tl.global_flag['mode']), network=net_d)

    ###============================= LOAD VGG ===============================###
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted( npz.items() ):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)
    # net_vgg.print_params(False)
    # net_vgg.print_layers()

    ###============================= TRAINING ===============================###
    ## use first `batch_size` of train set to have a quick test during training
    sample_imgs_HR = train_hr_imgs[0:batch_size]
    sample_imgs_LR = train_lr_imgs[0:batch_size]
    # sample_imgs = read_all_imgs(train_hr_img_list[0:batch_size], path=config.TRAIN.hr_img_path, n_threads=32) # if no pre-load train set
    sample_imgs_256 = tl.prepro.threading_data(sample_imgs_HR, fn=normalization)#crop_sub_imgs_fn, is_random=True)
    print('sample HR sub-image:',sample_imgs_256.shape, sample_imgs_256.min(), sample_imgs_256.max())
    sample_imgs_64 = tl.prepro.threading_data(sample_imgs_LR, fn=normalization)
    print('sample LR sub-image:', sample_imgs_64.shape, sample_imgs_64.min(), sample_imgs_64.max())
    tl.vis.save_images(sample_imgs_64, [ni, ni],save_dir_ginit+'/_train_sample_64.tif')
    tl.vis.save_images(sample_imgs_256, [ni, ni], save_dir_ginit+'/_train_sample_256.tif')
    tl.vis.save_images(sample_imgs_64, [ni, ni], save_dir_gan+'/_train_sample_64.tif')
    tl.vis.save_images(sample_imgs_256, [ni, ni], save_dir_gan+'/_train_sample_256.tif')

    ###========================= initialize G ====================###

    ## fixed learning rate
    MSE=[]
    sess.run(tf.assign(lr_v, lr_init))
    print(" ** fixed learning rate: %f (for init G)" % lr_init)
    for epoch in range(1, n_epoch_init+1):
        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0


        for idx in range(1, int(len(train_hr_imgs)/batch_size)+1):
            step_time = time.time()
            b_imgs_256 = tl.prepro.threading_data(train_hr_imgs[idx : idx + batch_size],fn=normalization)#crop_sub_imgs_fn, is_random=True)
            b_imgs_64 = tl.prepro.threading_data(train_lr_imgs[idx : idx + batch_size],fn=normalization)
            ## update G
            errM, _ = sess.run([mse_loss, g_optim_init], {t_image: b_imgs_64, t_target_image: b_imgs_256})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
            total_mse_loss += errM
            n_iter += 1
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss/n_iter)
        print(log)

        Epoch1.append(epoch)
        MSE.append(total_mse_loss)

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 5 == 0):
            out = sess.run(net_g_test.outputs, {t_image: sample_imgs_64})#; print('gen sub-image:', out.shape, out.min(), out.max())
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_ginit+'/train_%d.tif' % epoch)

        ## save model
        if (epoch != 0) and (epoch % 5 == 0):
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir+'/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)
        if(epoch==config.TRAIN.n_epoch_init):
            Arr1 = np.array(Epoch1)
            Arr2 = np.array(MSE)
            plt.plot(Arr1, Arr2)
            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            plt.title('MSE loss')
            plt.savefig(save_dir_ginit+'/Mse.png')


    ###========================= train GAN (SRGAN) =========================###

    # clipping method
    # clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, self.clip_values[0], self.clip_values[1])) for
    #                                      var in self.discriminator_variables]
    D_loss = []
    G_loss = []
    for epoch in range(1,n_epoch+1):
        ## update learning rate
        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter = 0, 0, 0

        ## If your machine have enough memory, please pre-load the whole train set.
        for idx in range(1, int(len(train_hr_imgs)/batch_size)+1):
            step_time = time.time()
            b_imgs_256 = tl.prepro.threading_data(train_hr_imgs[idx : idx + batch_size],fn=normalization)#crop_sub_imgs_fn, is_random=True)
            b_imgs_64 = tl.prepro.threading_data(train_lr_imgs[idx : idx + batch_size],fn=normalization)
            ## update D

            errD, summary,_, _ = sess.run([d_loss, merged, d_optim, clip_D], {t_image: b_imgs_64, t_target_image: b_imgs_256})
            loss_writer.add_summary(summary, idx)
            # d_vars = sess.run(clip_discriminator_var_op)
            ## update G
            errG, errM, errV, errA, _ = sess.run([g_loss, mse_loss, vgg_loss, g_gan_loss, g_optim],{t_image: b_imgs_64, t_target_image: b_imgs_256})

            print("Epoch [%2d/%2d] %4d time: %4.4fs, W_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)"
                  % (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errV, errA))
            total_d_loss += errD
            total_g_loss += errG
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss/n_iter, total_g_loss/n_iter)
        print(log)
        Epoch2.append(epoch)
        D_loss.append(total_d_loss)
        G_loss.append(total_g_loss)

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 5 == 0):
            out = sess.run(net_g_test.outputs, {t_image: sample_imgs_64})#; print('gen sub-image:', out.shape, out.min(), out.max())
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_gan+'/train_%d.tif' % epoch)

        ## save model
        if (epoch != 0) and (epoch % 5 == 0):
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir+'/g_{}.npz'.format(tl.global_flag['mode']), sess=sess)
            tl.files.save_npz(net_d.all_params, name=checkpoint_dir+'/d_{}.npz'.format(tl.global_flag['mode']), sess=sess)

        if (epoch ==config.TRAIN.n_epoch):
            Arr3=np.array(Epoch2)
            Arr4=np.array(G_loss)
            Arr5=np.array(D_loss)

            plt.plot(Arr3,Arr4)
            plt.xlabel('Epoch')
            plt.ylabel('G_loss')
            plt.title('Generator Loss')
            plt.savefig(save_dir_gan + '/Generator Loss.png')

            plt.plot(Arr3,Arr5)
            plt.xlabel('Epoch')
            plt.ylabel('D_loss')
            plt.title('Discriminator Loss')
            plt.savefig(save_dir_gan+'/Discriminator Loss.png')

def evaluate():
    ## create folders to save result images
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.tif', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.tif', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()


    ###========================== DEFINE MODEL ============================###
    imid =18
    #valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
    valid_lr_img = (valid_lr_imgs[imid] / (255./2.)) - 1  # rescale to ［－1, 1]
    valid_hr_img = (valid_hr_imgs[imid] / (255./2.)) - 1
    # print(valid_lr_img.min(), valid_lr_img.max())

    size = valid_lr_img.shape

    t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')

    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_srgan.npz', network=net_g)

    ###======================= EVALUATION =============================###
    start_time = time.time()
    out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
    print("took: %4.4fs" % (time.time() - start_time))

    print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("[*] save images")
    tl.vis.save_image(out[0], save_dir + '/valid_gen' + str(imid) + '.tif')
    tl.vis.save_image(valid_lr_img, save_dir + '/valid_lr' + str(imid) + '.tif')
    tl.vis.save_image(valid_hr_img, save_dir + '/valid_hr' + str(imid) + '.tif')

    out_bicu = scipy.misc.imresize(valid_lr_img, [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
    tl.vis.save_image(out_bicu, save_dir + '/valid_bicubic' + str(imid) + '.tif')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='evaluate(train)', help='srgan, evaluate(trian),evaluate(SZ), evaluate(XJ), evaluate(cross), cross, classfication(img), classfication(label)')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan':
        train()
    elif tl.global_flag['mode'] == 'evaluate(train)':
        evaluate()
    elif tl.global_flag['mode'] == 'evaluate(SZ)':
        evaluate()
    elif tl.global_flag['mode'] == 'evaluate(XJ)':
        evaluate()
    elif tl.global_flag['mode'] == 'evaluate(cross)':
        evaluate()
    elif tl.global_flag['mode'] == 'cross':
        evaluate()
    elif tl.global_flag['mode'] == 'classfication(img)':
        evaluate()
    elif tl.global_flag['mode'] == 'classfication(label)':
        evaluate()
    else:
        raise Exception("Unknow --mode")

import tensorflow as tf
import numpy as np
import math

import sys

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from pylab import *
from utils import *
from ops import batch_norm, linear, conv2d, deconv2d, lrelu
from tqdm import tqdm
from glob import glob
import os
import time


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

class StegoNet(object):
    def __init__(self, sess, a=0.2, b=0.4, c=0.4, msg_len=32, image_size=108, is_grayscale=False,
                 is_crop=True, output_size=64, batch_size=128,
                 epochs=2000, learning_rate=0.0001, train_prct=0.001, datapath='', savepath=''):
        """
        Args:
            sess: TensorFlow session
            See main.py for others
        """

        self.sess = sess
        self.a = a
        self.b = b
        self.c = c
        self.msg_len = msg_len
        self.batch_size = batch_size
        self.C_shp = [output_size, output_size, 3]
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.image_size = image_size
        self.train_prct = train_prct
        self.is_grayscale = is_grayscale
        self.output_size = output_size
        self.is_crop = is_crop
        self.datapath = datapath
        self.savepath = savepath

        print( "a: %.2f, b: %.2f, c: %.2f" %(self.a, self.b, self.c))

    def WriteToFile(self, fp, src):
        if not os.path.exists(fp):
            open(fp, 'w').close()
        with open(fp, mode='a') as file:
            file.write('%s\n' % (src))

    def alice_model(self, data_input_image = None, data_input_msg = None):

        s_h, s_w = self.output_size, self.output_size
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        ####### Alice's network #######
        # Alice's input
        self.alice_input_image = data_input_image
        self.alice_input_msg = data_input_msg
        self.alice_input = tf.concat([tf.reshape( self.alice_input_image, [self.batch_size, -1] ), self.alice_input_msg], 1)

        ## CNN
        self.alice0 = linear(self.alice_input, self.output_size*8*s_h16*s_w16, 'alice0')
        self.alice1 = tf.reshape(self.alice0, [-1, s_h16, s_w16, self.output_size * 8])
        self.alice_bn1 = batch_norm(name='alice_bn1')
        alice1 = tf.nn.relu(self.alice_bn1(self.alice1))

        self.alice2 = deconv2d(alice1, [self.batch_size, s_h8, s_w8, self.output_size * 4], name='alice2')
        self.alice_bn2 = batch_norm(name='alice_bn2')
        alice2 = tf.nn.relu(self.alice_bn2(self.alice2))

        self.alice3 = deconv2d(alice2, [self.batch_size, s_h4, s_w4, self.output_size * 2], name='alice3')
        self.alice_bn3 = batch_norm(name='alice_bn3')
        alice3 = tf.nn.relu(self.alice_bn3(self.alice3))

        self.alice4 = deconv2d(alice3, [self.batch_size, s_h2, s_w2, self.output_size * 1], name='alice4')
        self.alice_bn4 = batch_norm(name='alice_bn4')
        alice4 = tf.nn.relu(self.alice_bn4(self.alice4))

        self.alice5 = deconv2d(alice4, [self.batch_size, s_h, s_w, 3], name='alice5')
        return tf.nn.tanh(self.alice5)

    def bob_model(self, data_input_image = None):

        ####### Bob's network #######

        # bob's input
        self.bob_input = data_input_image
        print( self.bob_input)

        self.bob0 = lrelu(conv2d(self.bob_input, self.output_size, name='bob_h0_conv'))
        self.bob_bn1 = batch_norm(name='bob_bn1')
        self.bob1 = lrelu(self.bob_bn1(conv2d(self.bob0, self.output_size*2, name='bob_h1_conv')))
        self.bob_bn2 = batch_norm(name='bob_bn2')
        self.bob2 = lrelu(self.bob_bn2(conv2d(self.bob1, self.output_size*4, name='bob_h2_conv')))
        self.bob_bn3 = batch_norm(name='bob_bn3')
        self.bob3 = lrelu(self.bob_bn3(conv2d(self.bob2, self.output_size*8, name='bob_h3_conv')))
        self.bob4 = linear(tf.reshape(self.bob3, [self.batch_size, -1]), self.msg_len, 'bob_h3_lin')
        return tf.nn.tanh(self.bob4)

    def eve_model(self, data_input = None, reuse=False):

        ####### Eve's network #######

        with tf.variable_scope("eve") as scope:
            if reuse:
                scope.reuse_variables()

            self.eve_input = data_input

            print( self.eve_input)

            self.eve0 = lrelu(conv2d(self.eve_input, self.output_size, name='eve_h0_conv'))
            self.eve_bn1 = batch_norm(name='eve_bn1')
            self.eve1 = lrelu(self.eve_bn1(conv2d(self.eve0, self.output_size*2, name='eve_h1_conv')))
            self.eve_bn2 = batch_norm(name='eve_bn2')
            self.eve2 = lrelu(self.eve_bn2(conv2d(self.eve1, self.output_size*4, name='eve_h2_conv')))
            self.eve_bn3 = batch_norm(name='eve_bn3')
            self.eve3 = lrelu(self.eve_bn3(conv2d(self.eve2, self.output_size*8, name='eve_h3_conv')))
            self.eve4 = linear(tf.reshape(self.eve0, [self.batch_size, -1]), 1, 'eve_h3_lin')
            return self.eve4, self.eve4

    def batch_data_paths(self):
        data_paths = glob(os.path.join(self.datapath, "*.jpg"))
        data_paths = np.array(data_paths)
        num_imgs = len(data_paths)
        np.random.seed(35)
        np.random.shuffle(data_paths)
        self.num_to_train = int(math.ceil(num_imgs * self.train_prct))
        self.num_batches = int(math.floor(self.num_to_train / float(self.batch_size)))
        self.num_to_train = int(math.ceil(self.num_batches * self.batch_size))
        data_paths = data_paths[:self.num_to_train]
        batched_data_paths = np.reshape(data_paths, (self.num_batches, self.batch_size))
        self.WriteToFile(self.savepath + "stats.txt", "Number of data samples to train: %s out of %s" %((self.num_to_train), num_imgs))
        return batched_data_paths

    def load_data(self, data_path_batch):

        data = [get_image(batch_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_size, is_grayscale = self.is_grayscale) for batch_file in data_path_batch]
        if (self.is_grayscale):
            data_images = np.array(data).astype(np.float32)[:, :, :, None]
        else:
            data_images = np.array(data).astype(np.float32)

        data_images = np.reshape(data_images, (self.batch_size, data_images.shape[1], data_images.shape[2], data_images.shape[3]))
        
        return data_images

    def train(self):

        # Placeholder variables for cover image (C), noise (that is converted by alice to an image),
        self.C = tf.placeholder(tf.float32, shape = [self.batch_size] + self.C_shp, name='cover_img')
        self.msg = tf.placeholder(tf.float32, shape = [self.batch_size, self.msg_len], name='message_string')

        self.alice_encode = self.alice_model( data_input_image = self.C, data_input_msg = self.msg )
        self.bob_decode = self.bob_model( data_input_image = self.alice_encode )
        self.eve_real_images, self.eve_real_images_logits = self.eve_model( data_input = self.C )
        self.eve_steg_images, self.eve_steg_images_logits = self.eve_model( data_input = self.alice_encode, reuse=True)

        self.eve_loss_real = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.eve_real_images_logits, labels=tf.ones_like(self.eve_real_images)))

        self.eve_loss_steg = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.eve_steg_images_logits, labels=tf.zeros_like(self.eve_steg_images)))

        self.eve_loss = self.eve_loss_real + self.eve_loss_steg

        self.bob_loss = tf.reduce_mean(tf.pow(self.msg - self.bob_decode, 2)) #l2

        self.alice_bob_loss = ( self.a*tf.reduce_mean(
                                    tf.nn.sigmoid_cross_entropy_with_logits(
                                    logits=self.eve_steg_images_logits, labels=tf.ones_like(self.eve_steg_images))) + \
                                self.b*tf.reduce_mean(tf.abs(self.alice_encode - self.C))  +\
                                self.c*tf.reduce_mean(tf.pow(self.msg - self.bob_decode, 2)) )

        # Get training variables corresponding to each network
        self.t_vars = tf.trainable_variables()
        self.alice_or_bob_vars = [var for var in self.t_vars if 'alice' in var.name or 'bob' in var.name]
        self.eve_vars = [var for var in self.t_vars if 'eve' in var.name]

        # Build the optimizers
        self.alice_bob_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.alice_bob_loss, var_list=self.alice_or_bob_vars)
        self.eve_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.eve_loss, var_list=self.eve_vars)

        ## Load celebA dataset ##
        print("Loading data paths..")
        batched_data_paths = self.batch_data_paths()
        print("Finished loading data paths..")

        # Begin Training
        tf.global_variables_initializer().run()
        counter = 1
        start_time = time.time()

        batch_num_paths = batched_data_paths[0]
        cover_images = self.load_data(batch_num_paths)
        self.WriteToFile(self.savepath + "stats.txt", "Data shape: %s" %(' '.join(str(x) for x in cover_images.shape)))
        self.plot_generated_images(cover_images, 'real_output', self.output_size)
        self.num_correct_bits = [ ]
        self.eve_errors = [ ]
        self.alice_errors = [ ]
        self.bob_errors = [ ]

        for e in range(self.epochs):

            for i in range(batched_data_paths.shape[0]):

                msg = (np.random.randint(0, 2, size=(self.batch_size, self.msg_len))*2-1)/2.
                batch_num_paths = batched_data_paths[i]
                cover_images = self.load_data(batch_num_paths)

                # train eve
                _, decrypt_err_eve = self.sess.run([self.eve_optimizer, self.eve_loss],
                                               feed_dict={self.C: cover_images, self.msg: msg})

                # train alice/bob -- train them more than eve (not implemented)
                for _ in range(1):
                    _, decrypt_err_alice_bob = self.sess.run([self.alice_bob_optimizer, self.bob_loss],
                                               feed_dict={self.C: cover_images, self.msg: msg})

                err_eve_steg = self.eve_loss_steg.eval({ self.C: cover_images, self.msg: msg })
                err_eve_real = self.eve_loss_real.eval({ self.C: cover_images })
                err_alice = self.alice_bob_loss.eval({ self.C: cover_images, self.msg: msg })
                err_bob = self.bob_loss.eval({ self.C: cover_images, self.msg: msg })

                counter += 1

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, alice_loss: %.8f, bob_loss: %.8f, eve_steg_loss: %.8f, eve_real_loss: %.8f" \
                  % (e, i, batched_data_paths.shape[0],
                    time.time() - start_time, err_alice, err_bob, err_eve_steg, err_eve_real))

                if i==0 and e%500 == 0:
                    generated_images = self.sess.run(self.alice_encode, feed_dict={self.C: cover_images, self.msg: msg})
                    self.plot_generated_images(generated_images, 'noise_output', e)



            bob_decoded = self.sess.run(self.bob_decode, feed_dict={self.C: cover_images, self.msg: msg})
            correct_bits = np.mean([sum([1 for i in range(len(bob_decoded[j])) if np.floor(bob_decoded)[j][i]==np.floor(msg)[j][i]]) for j in range(len(bob_decoded))])
            print("Correct decoded bits: %.2f out of %.2f" %(correct_bits, self.msg_len))
            self.num_correct_bits.append(correct_bits)
            self.eve_errors.append(err_eve_steg + err_eve_real)            
            self.alice_errors.append(err_alice)            
            self.bob_errors.append(err_bob)            
            
            self.WriteToFile(self.savepath + "stats.txt", "Epoch %d - Bob Decoded shape %s" %(e, ' '.join(str(x) for x in bob_decoded.shape)))
            self.WriteToFile(self.savepath + "stats.txt", "Epoch %d - Bob Decoded[0] %s" %(e, bob_decoded[0]))
            self.WriteToFile(self.savepath + "stats.txt", "Epoch %d - Msg shape %s" %(e, ' '.join(str(x) for x in msg.shape)))
            self.WriteToFile(self.savepath + "stats.txt", "Epoch %d - Msg Decoded[0] %s" %(e, msg[0]))
            self.WriteToFile(self.savepath + "stats.txt", "Epoch %d - Correct decoded bits %s" %(e, correct_bits))
            self.WriteToFile(self.savepath + "Training_Errors.txt", "Epoch %d - Alice %s" %(e, err_alice))
            self.WriteToFile(self.savepath + "Training_Errors.txt", "Epoch %d - Bob %s" %(e, err_bob))
            self.WriteToFile(self.savepath + "Training_Errors.txt", "Epoch %d - Eve %s" %(e, err_eve_steg+err_eve_real))

        self.plot_errors()
        self.plot_correct_bits()

    def plot_correct_bits(self):
        """
        Plot the number of correct bits decoded by Bob
        """
        plt.clf()
        #sns.set_style("darkgrid")
        plt.plot(range(self.epochs), self.num_correct_bits)
        plt.xlabel('Epochs')
        plt.ylabel('Number of bits correctly decoded (out of %.2f)' %(self.msg_len))
        plt.axis('tight')
        plt.savefig(self.savepath + "correct_bits.pdf", format="pdf", transparent=True, bbox_inches='tight',)

    def plot_errors(self):
        """
        Plot Lowest Decryption Errors achieved by Bob and Eve per epoch
        """
        plt.clf()
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = fig.add_subplot(111)
        ax3 = fig.add_subplot(111)
        ax1.plot(self.alice_errors, color='b', linestyle='-', label='Alice')
        ax2.plot(self.bob_errors, color='g', linestyle='--', label='Bob')
        ax3.plot(self.eve_errors, color='r', linestyle=':', label='Eve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.axis('tight')
        plt.savefig(self.savepath + "eve_loss.pdf", format="pdf", transparent=True, bbox_inches='tight',)

    def plot_generated_images(self, images, network, epoch):
        plt.clf()
        #sns.set_style("darkgrid")
        for i, img in enumerate(images[:9]):
            i = i+1
            plt.subplot(3, 3, i)
            img = (img + 1)*127.5
            plt.imshow(img.astype(np.uint8))
            plt.axis('off')
        plt.savefig('./results/' + str(network) + '_output_' + str(epoch) + '.pdf', format="pdf", transparent=True, bbox_inches='tight',)

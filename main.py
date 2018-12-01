import tensorflow as tf

import os

from argparse import ArgumentParser
from train import StegoNet


def build_parser():
    parser = ArgumentParser()

    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=0.0002)

    parser.add_argument('--datapath', type=str,
                        dest='datapath', help='Path to CelebrityA dataset',
                        metavar='DATAPATH', default='./data/')

    parser.add_argument('--savepath', type=str,
                        dest='savepath', help='Where to save images and training metrics',
                        metavar='SAVEPATH', default='./results/')

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='Number of Epochs in Adversarial Training',
                        metavar='EPOCHS', default=501)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=32)

    parser.add_argument('--image-size', type=int,
                        dest='image_size', help='The size of image to use (will be center cropped) [108]',
                        metavar='IMAGE_SIZE', default=108)

    parser.add_argument('--train-prct', type=float,
                        dest='train_prct', help='The fraction of images used for training',
                        metavar='TRAIN_PRCT', default=0.01)

    parser.add_argument('--is-grayscale', type=bool,
                        dest='is_grayscale', help='Is the dataset in grayscale or not',
                        metavar='IS_GRAYSCALE', default=False)

    parser.add_argument('--is-crop', type=bool,
                        dest='is_crop', help='Is the image cropped',
                        metavar='IS_CROP', default=True)

    parser.add_argument('--output-size', type=bool,
                        dest='output_size', help='The size of the output images to produce [64]',
                        metavar='OUTPUT_SIZE', default=32)

    parser.add_argument('--msg-len', type=int,
                        dest='msg_len', help='The size of the message input to Alice / output by Bob',
                        metavar='MSG_LEN', default=100)

    parser.add_argument('--a', type=float,
                        dest='a', help='The amount of weight put on eve loss',
                        metavar='A', default=0.1)

    parser.add_argument('--b', type=float,
                        dest='b', help='The amount of weight put on alice reconstruction loss',
                        metavar='B', default=0.3)

    parser.add_argument('--c', type=float,
                        dest='c', help='The amount of weight put on bob message reconstruction loss',
                        metavar='C', default=0.6)

    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    if not os.path.exists(options.savepath):
        os.mkdir(options.savepath)

    with tf.Session() as sess:
        stego_net = StegoNet(sess, epochs=options.epochs, a=options.a, b=options.b, c=options.c,
                               batch_size=options.batch_size, learning_rate=options.learning_rate,
                               msg_len=options.msg_len, image_size=options.image_size, is_grayscale=options.is_grayscale,
                               is_crop=options.is_crop, output_size=options.output_size, train_prct=options.train_prct,
                               datapath=options.datapath, savepath=options.savepath)

        stego_net.train()

if __name__ == '__main__':
    main()

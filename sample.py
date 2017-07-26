from __future__ import print_function

import argparse
import os

import tensorflow as tf
from six import text_type
from six.moves import cPickle

from model import Model


def main():
	parser = argparse.ArgumentParser(
					   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--save_dir', type=str, default='save',
						help='model directory to store checkpointed models')
	parser.add_argument('-n', type=int, default=500,
						help='number of characters to sample')
	parser.add_argument('--prime', type=text_type, default=u' ',
						help='prime text')
	parser.add_argument('--sample', type=int, default=1,
						help='0 to use max at each timestep, 1 to sample at '
							 'each timestep, 2 to sample on spaces')

	args = parser.parse_args()
	sample(args)


def sample(args):
	with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
		saved_args = cPickle.load(f)
	with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
		chars, vocab = cPickle.load(f)

	saved_args.batch_size = 1
	saved_args.seq_length = 1

	model = Model(saved_args, args.n, training=False)

	print("Sample model built")

	os.environ["CUDA_VISIBLE_DEVICES"]="0"

	with tf.Session() as sess:
		tf.global_variables_initializer().run()

		saver = tf.train.Saver(tf.global_variables())
		print("Got saver foir trainable variables")

		ckpt = tf.train.get_checkpoint_state(args.save_dir)
		print("Loaded checkpoint from save dir")
		if ckpt and ckpt.model_checkpoint_path:
			print("Restoring...")
			saver.restore(sess, ckpt.model_checkpoint_path)
			print("Restored checkpoint successfully")
			stuff = model.sample(sess, chars, vocab, args.n, 
					args.prime, args.sample).encode("utf-8")
			print("BEGIN RESULT")
			print(stuff)
			print("END RESULT")

if __name__ == '__main__':
	main()

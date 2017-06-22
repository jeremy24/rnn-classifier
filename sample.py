from __future__ import print_function
import tensorflow as tf

import argparse
import os
from six.moves import cPickle

from model import Model

from six import text_type


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
	model = Model(saved_args, args.n, training=False)

	print("Sample model built")

	

	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		
		print("Globals:")
		#for x in tf.global_variables():
		#	print(x)
		#exit(1)

		saver = tf.train.Saver(tf.global_variables()[5:])
		print("Got saver")
		ckpt = tf.train.get_checkpoint_state(args.save_dir)
		print("Loaded checkpoint")
		if ckpt and ckpt.model_checkpoint_path:
			print("Restoring...")
			saver.restore(sess, ckpt.model_checkpoint_path)
			print("Restored checkpoint")
			print(model.sample(sess, chars, vocab, args.n, args.prime,
							   args.sample).encode('utf-8'))

if __name__ == '__main__':
	main()

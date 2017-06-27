from __future__ import print_function
import tensorflow as tf

import argparse
import os
from six.moves import cPickle

from model import Model

from six import text_type

from utils import TextLoader

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


def run_test(sess, model, data, state):
	x = data[0]
	y = data[1]
	
	try:
		print("x shape: ", x.shape, "y shape: ", y.shape)

		feed = { model.input_data: x, model.initial_state: state }

		probs, state = sess.run([model.probs, model.final_state], feed)
		y_ = probs[0]

		correct = tf.equal(tf.argmax(y), tf.argmax(y_))
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
	
		acc = sess.run(accuracy, feed_dict={y: y, y_: y_})
	
		print("Accuracy: ", acc)

	except Exception as ex:
		print("run_test: ", ex)
		exit(1)

def sample(args):
	with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
		saved_args = cPickle.load(f)
	with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
		chars, vocab = cPickle.load(f)

	saved_args.batch_size = 1
	saved_args.seq_length = 1

	data_loader = TextLoader(saved_args.data_dir, saved_args.save_dir, saved_args.batch_size, 
			saved_args.seq_length)
	
	print("Data loaded in")

	args.vocab_size = data_loader.vocab_size

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
			
			
			
			i = 0
			state = sess.run(model.cell.zero_state(1, tf.float32))
			batches = data_loader.test_batches

			for item in batches:
				x = item[0]
				y = item[1]
				run_test(sess, model, item, state)


			
			stuff = model.sample(sess, chars, vocab, args.n, 
					args.prime, args.sample).encode("utf-8")
			print("BEGIN RESULT")
			print(stuff)
			print("END RESULT")

if __name__ == '__main__':
	main()

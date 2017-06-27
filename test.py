from __future__ import print_function
import tensorflow as tf

import argparse
import os
from six.moves import cPickle

from model import Model

from six import text_type
from utils import TextLoader

import numpy as np

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





def run_test(sess, model, batch, state):
	
	# both are => [ batch_size, seq_length ]
	x_seq = batch[0]
	y_seq = batch[1]

	
	try:
		#print("x_seq shape: ", x_seq.shape, "y_seq shape: ", y_seq.shape)
		
		feed = { model.input_data: x_seq, model.initial_state: state }
		probs, state = sess.run([model.probs, model.final_state], feed)
		
		y_seq_ = probs
	
		#print("y_seq_ shape: ", y_seq_.shape)

		y_ = list()

		for item in y_seq_:
			a = list()
			for sub in item:
				a.append(np.argmax(sub))
			y_.append(a)

		y_ = np.array(y_)
		
		#print("y_ shape: ", y_.shape)
		
		correct = np.equal(y_seq, y_)
		accuracy = np.mean(correct)

		return accuracy, state

	except Exception as ex:
		print("run_test: ", ex)
		exit(1)

def sample(args):
	with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
		saved_args = cPickle.load(f)
	with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
		chars, vocab = cPickle.load(f)

	

	# saved_args.batch_size = 1
	# saved_args.seq_length = 1

	print("Saved args: ", saved_args)

	data_loader = TextLoader(saved_args.data_dir, saved_args.save_dir, saved_args.batch_size, 
			saved_args.seq_length)
	
	print("Data loaded in")

	args.vocab_size = data_loader.vocab_size

	model = Model(saved_args, 1, training=False)

	print("Sample model built")
	

	os.environ["CUDA_VISIBLE_DEVICES"]="0"

	tests_to_run = 100

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
			state = sess.run(model.cell.zero_state(saved_args.batch_size, tf.float32))
			
			print("\ngot initial state")

			batches = data_loader.batches

			print("\nGrabbed batches")
			
			total_accuracy = 0.0

			print("Running", tests_to_run, "test batches")

			for batch in batches:
				if i == tests_to_run:
					break
				# batch => [2]     [x, y] pairs of data where
				# x and y are => [ batch_size, seq_length ]
				if i % 100 == 0:
					print("On batch:", i)
				accuracy, _ = run_test(sess, model, batch, state)
				total_accuracy += accuracy
				state = sess.run(model.cell.zero_state(saved_args.batch_size, tf.float32))
				i += 1

			total_accuracy = total_accuracy / 1
			total_accuracy = round(total_accuracy, 4)

			print("Accuracy: {}% over {} batches".format(total_accuracy, i))
					

if __name__ == '__main__':
	main()

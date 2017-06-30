""" test a tensorflow model """

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import os

from six.moves import cPickle

import numpy as np
import tensorflow as tf
import sklearn.metrics as skmetrics

# my stuff
from model import Model
from utils import TextLoader


def main():
	"""main stuff"""
	parser = argparse.ArgumentParser(
					   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--save_dir', type=str, default='save',
						help='model directory to load from')
	parser.add_argument('-n', type=int, default=100,
						help='number of test batches to sample')
	parser.add_argument('--sample', type=int, default=1,
						help='0 to use max at each timestep, 1 to sample at '
							 'each timestep, 2 to sample on spaces')

	args = parser.parse_args()
	test(args)





def run_test(sess, model, x_seq, y_seq, state):
	""" run a test for a single batch """
	# both are => [ batch_size, seq_length ]
	# x_seq = batch[0]
	# y_seq = batch[1]

	
	try:
		#print("x_seq shape: ", x_seq.shape, "y_seq shape: ", y_seq.shape)
		
		feed = { model.input_data: x_seq, model.targets: y_seq,  model.initial_state: state }
		probs, state, loss = sess.run([model.probs, model.final_state, model.cost], feed)
		
		y_seq_ = probs
	
		#print("y_seq_ shape: ", y_seq_.shape)

		y_bar = list()

		for item in y_seq_:
			sublist = list()
			for sub in item:
				sublist.append(np.argmax(sub))
			y_bar.append(sublist)

		y_bar = np.array(y_bar)
		# accuracy = np.mean(np.equal(y_seq, y_bar))


		try:		
			y_seq = np.ndarray.flatten(y_seq)
			y_bar = np.ndarray.flatten(y_bar)
			confusion = skmetrics.confusion_matrix(y_seq, y_bar)
		except ValueError as ex:
			print("Error build confusion:", ex)
			print(y_seq.shape)
			print("labels y_bar:", set(y_bar))
			print("labels y_seq:", set(y_seq))
			exit(1)

		tn = confusion[0,0]
		fp = confusion[0,1]
		fn = confusion[1,0]
		tp = confusion[1,1]
	

		precision = tp / (fp + tp)
		recall = tp / (tp+fn)
		accuracy = (tn + tp) / (tn + fp + fn + tp)
		sensitivity = recall # same thing
		specificity = tn / (tn + fp)

	

		ret = dict()
		ret["accuracy"] = accuracy
		# ret["state"] = state
		ret["loss"] = loss
		ret["precision"] = precision
		ret["recall"] = recall
		ret["sensitivity"] = sensitivity
		ret["specificity"] = specificity

		print(tp, fp, tn, fn)
		print(tp, "/", "(", fp, "+", tp, ")")
		# print(ret)

		return ret

	except Exception as ex:
		print("run_test: ", ex)
		exit(1)

def test(args):
	""" run the tests """
	saved_args = None
	data_loader = None
	model = None

	with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as fin:
		saved_args = cPickle.load(fin)


	print("Saved args: ", saved_args)

	data_loader = TextLoader(saved_args.data_dir, saved_args.save_dir, saved_args.batch_size, 
			saved_args.seq_length)

	data_loader.batches = data_loader.test_batches
	data_loader.num_batches = len(data_loader.batches)

	print("Data loaded in")

	args.vocab_size = data_loader.vocab_size

	model = Model(saved_args, 1, training=False)

	print("Sample model built")
	

	os.environ["CUDA_VISIBLE_DEVICES"]="0"

	
	def t_print(stuff):
		"""pretty print stats from a list"""
		print("\tAvg: {:.3f}".format(np.mean(stuff)))
		print("\tMax: {:.3f}".format(np.max(stuff)))
		print("\tMin: {:.3f}".format(np.min(stuff)))
		print("\tStd: {:.3f}".format(np.std(stuff)))
	

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
			
			
			state = sess.run(model.cell.zero_state(saved_args.batch_size, tf.float32))
			
			print("\ngot initial state")
			print("\nGrabbed batches")
			print("Running", args.n, "test batches")
			
			losses = list()	
			accs = list()
			precs = list()
			recalls = list()
			i = 0

			for batch in data_loader.batches:
				# state = sess.run(model.cell.zero_state(saved_args.batch_size, tf.float32))
				if i == args.n: # number to run
					break
				# batch => [2]	   [x, y] pairs of data where
				# x and y are => [ batch_size, seq_length ]
				if i % 100 == 0:
					print("On batch:", i)
				#accuracy, _, loss = run_test(sess, model, batch[0], batch[1], state)
				x = batch[0]
				y = batch[1]
				metrics = run_test(sess, model, x, y, state)
				
				losses.append(metrics["loss"])
				accs.append(metrics["accuracy"])
				precs.append(metrics["precision"])
				recalls.append(metrics["recall"])
				i += 1
			
			print("\nFor {} batches".format(i))
			print("Accuracy:")
			t_print(accs)
			
			print("Loss:")
			t_print(losses)
			
			print("Precision:")
			t_print(precs)

			print("Recall:")
			t_print(recalls)

if __name__ == '__main__':
	main()

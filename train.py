""" the model """
from __future__ import print_function

import argparse
import time
import os
import json
import threading
from six.moves import cPickle

from utils import TextLoader
from model import Model
import numpy as np

import tensorflow as tf
from tensorflow.python.client import timeline

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/extras/CUPTI/lib64/"


def dump_args(args):
	"""dump args to a file"""
	try:
		filename = os.path.join(args.save_dir, "hyper_params.json")
		with open(filename, "w+") as fout:
			data = dict()
			args = vars(args)
			for key in args:
				data[key] = args[key]
			fout.write(json.dumps(data, sort_keys=True, indent=4, separators=(",", ":")))
			fout.close()
	except Exception as ex:
		print("Unable to save args to file: ", ex.message)
		exit(1)


def main():
	"""main fn"""
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
						help='data directory containing input.txt')
	parser.add_argument('--save_dir', type=str, default='save',
						help='directory to store checkpointed models')
	parser.add_argument('--log_dir', type=str, default='logs',
						help='directory to store tensorboard logs')
	parser.add_argument('--rnn_size', type=int, default=128,
						help='size of RNN hidden state')
	parser.add_argument('--num_layers', type=int, default=2,
						help='number of layers in the RNN')
	parser.add_argument('--model', type=str, default='lstm',
						help='rnn, gru, lstm, or nas')
	parser.add_argument('--batch_size', type=int, default=50,
						help='minibatch size')
	parser.add_argument('--seq_length', type=int, default=50,
						help='RNN sequence length')
	parser.add_argument('--num_epochs', type=int, default=50,
						help='number of epochs')
	parser.add_argument('--save_every', type=int, default=1000,
						help='save frequency')
	parser.add_argument('--grad_clip', type=float, default=5.,
						help='clip gradients at this value')
	parser.add_argument('--learning_rate', type=float, default=0.002,
						help='learning rate')
	parser.add_argument('--decay_rate', type=float, default=0.97,
						help='decay rate for rmsprop')
	parser.add_argument('--output_keep_prob', type=float, default=1.0,
						help='probability of keeping weights in the hidden layer')
	parser.add_argument('--input_keep_prob', type=float, default=1.0,
						help='probability of keeping weights in the input layer')
	parser.add_argument('--init_from', type=str, default=None,
						help="""continue training from saved model at this path. Path must contain files saved by previous training process:
							'config.pkl'		: configuration;
							'chars_vocab.pkl'	: vocabulary definitions;
							'checkpoint'		: paths to model file(s) (created by tf).
												  Note: this file contains absolute paths, be careful when moving files around;
							'model.ckpt-*'		: file(s) with model definition (created by tf)
						""")
	parser.add_argument("--gpu", type=str, default="1", help="Gpu[s] to make available to tf")
	parser.add_argument("--print_cycle", type=int, default=500, help="Cycle to print to console")
	args = parser.parse_args()
	dump_args(args)

	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

	train(args)


def dump_data(data_loader, args):
	print("Dumping out pickled data...")
	if not os.path.isdir(args.save_dir):
		os.makedirs(args.save_dir)
	with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
		cPickle.dump(args, f)
	with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
		cPickle.dump((data_loader.chars, data_loader.vocab), f)


def train(args):
	data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
	args.vocab_size = data_loader.vocab_size

	print("Vocab size: ", args.vocab_size)

	# check compatibility if training is continued from previously saved model
	if args.init_from is not None:
		# check if all necessary files exist
		assert os.path.isdir(args.init_from), " %s must be a a path" % args.init_from
		assert os.path.isfile(
			os.path.join(args.init_from, "config.pkl")), "config.pkl file does not exist in path %s" % args.init_from
		assert os.path.isfile(os.path.join(args.init_from,
										"chars_vocab.pkl")), "chars_vocab.pkl.pkl file does not exist in path %s" % args.init_from
		ckpt = tf.train.get_checkpoint_state(args.init_from)
		assert ckpt, "No checkpoint found"
		assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

		# open old config and check if models are compatible
		with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
			saved_model_args = cPickle.load(f)
		need_be_same = ["model", "rnn_size", "num_layers", "seq_length"]
		for checkme in need_be_same:
			assert vars(saved_model_args)[checkme] == vars(args)[
				checkme], "Command line argument and saved model disagree on '%s' " % checkme

		# open saved vocab/dict and check if vocabs/dicts are compatible
		with open(os.path.join(args.init_from, 'chars_vocab.pkl'), 'rb') as f:
			saved_chars, saved_vocab = cPickle.load(f)
		assert saved_chars == data_loader.chars, "Data and loaded model disagree on character set!"
		assert saved_vocab == data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"

	dump_data(data_loader, args)

	print_cycle = args.print_cycle
	total_steps = args.num_epochs * data_loader.num_batches

	print("Building model")
	model = Model(args, data_loader.num_batches)

	print("Model built")

	# refresh dumped data since some models
	# change the values of args
	dump_data(data_loader, model.args)
	dump_args(args)

	# used if you want a lot of logging
	sess_config = tf.ConfigProto(log_device_placement=False)
	
	# used to watch gpu memory thats actually used
	sess_config.gpu_options.allow_growth = True

	# set up some data capture lists
	global_start = time.time()

	args.data = dict()

	args.data["losses"] = list()
	args.data["avg_time_per_step"] = list()
	args.data["logged_time"] = list()

	with tf.Session( config = sess_config) as sess:
		# instrument for tensorboard
		summaries = tf.summary.merge_all()
		writer = tf.summary.FileWriter(
			os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))
		writer.add_graph(sess.graph)

		print("Saving global variables")
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())
		# restore model
		if args.init_from is not None:
			saver.restore(sess, ckpt.model_checkpoint_path)

		print("Starting...")
		print("Have {} epochs and {} batches per epoch".format(args.num_epochs, data_loader.num_batches))
		total_time = 0.0
		# run_meta = tf.RunMetadata()
		for epoch in range(args.num_epochs):
			sess.run(tf.assign(model.lr,
							args.learning_rate * (args.decay_rate ** epoch)))

			print("Resetting batch pointer for epoch: ", epoch)
			data_loader.reset_batch_pointer()
			# state = sess.run(model.initial_state)

			start = time.time()

			print("Copying over data for epoch")

			x_batches = list()
			y_batches = list()

			print("Breaking out the shuffled batches")
			for i in range(len(data_loader.batches)):
				x_batches.append(data_loader.batches[i][0])
				y_batches.append(data_loader.batches[i][1])

			print("Copying over all data")
			x_batches = np.array(x_batches, dtype=np.float32)
			y_batches = np.array(y_batches, dtype=np.float32)
			try:

				
				
				print("x_batches shape: ", x_batches.shape)			
				sess.run(tf.assign(model.all_input_data, x_batches))
				print("x_batches copied")
				sess.run(tf.assign(model.all_target_data, y_batches))
				print("y_batches copied")
				sess.run(tf.assign(model.step, 0))
				


				print("Step copied")
			except ValueError as valEr:
				print("Error copying over data: ", valEr)
				exit(1)

			print("Successfully copied over the data for epoch {}".format(epoch))

			for batch in range(data_loader.num_batches):

				step = epoch * data_loader.num_batches + batch

				# if printing
				if step % print_cycle == 0 and step > 0:
					summary, train_loss, state, _ = sess.run([summaries, model.cost, model.final_state, model.train_op])
					writer.add_summary(summary, step)
					end = time.time()

					total_time += end - start
					avg_time_per = round(total_time / step if step > 0 else step + 1, 2)
					steps_left = total_steps - step
					print("{}/{} (epoch {}), train_loss: {:.3f}, time/{}: {:.3f} time/step = {:.3f}  time left: {:.2f}m"
						.format(step, total_steps, epoch, train_loss, print_cycle,
						end - start, avg_time_per, steps_left * avg_time_per / 60))

					start = time.time()

					global_diff = time.time() - global_start
					args.data["losses"].append(float(train_loss))
					args.data["logged_time"].append(int(global_diff))
					args.data["avg_time_per_step"].append(float(avg_time_per))

				else:  # else normal training
					train_loss, state, _ = sess.run(
						[model.cost, model.final_state, model.train_op])

				last_batch = batch == data_loader.num_batches - 1
				last_epoch = epoch == args.num_epochs - 1
				if step % args.save_every == 0 or (last_batch and last_epoch):
					# save for the last result
					checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
					saver.save(sess, checkpoint_path, global_step=step)
					dump_args(args)
					print("model saved to {}".format(checkpoint_path))
					args.data["last_recorded_loss"] = {
						"time": int(time.time() - global_start),
						"loss": float(train_loss)
					}
					args.data["total_train_time"] = {
						"steps": int(total_steps),
						"time": int(time.time() - global_start)
					}
				# increment the model step
				sess.run(model.inc_step)


if __name__ == '__main__':
	main()

""" the model """
from __future__ import print_function

import argparse
import time
import os
import json
import threading
import math
import re
import gc
import traceback
from six.moves import cPickle

from labeler import labeler, LabelTypes
from data_loader import TextLoader
from model import Model
from data_blob import Prepositions
import numpy as np

import tensorflow as tf

from decorators import *

from tensorflow.python.client import timeline
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/extras/CUPTI/lib64/"


def dump_args(args):
	"""dump args to a file"""
	try:
		filename = os.path.join(args.save_dir, "hyper_params.json")
		with open(filename, "w") as fout:
			data = dict()
			args = vars(args)
			for key in args:
				data[key] = args[key]
			fout.write(json.dumps(data, sort_keys=True, indent=4, separators=(",", ":")))
			fout.close()
	except Exception as ex:
		print("Unable to save args to file: ", ex)
		exit(1)


def main():
	"""main fn"""
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
						help='data directory containing input.txt')
	parser.add_argument('--use_weights', type=int, default=1,
						help='Whether to weight the losses')
	parser.add_argument('--save_dir', type=str, default='save',
						help='directory to store model checkpoints')
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
	os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/extra/CUPTI/lib64/"

	print("Kicking off train call...")

	train(args)


def dump_data(data_loader, args):
	print("Dumping out pickled data...")
	if not os.path.isdir(args.save_dir):
		os.makedirs(args.save_dir)
	with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
		cPickle.dump(args, f)
	with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
		cPickle.dump((data_loader.chars, data_loader.vocab), f)


def pretty_print(item, step, total_steps, epoch, print_cycle, end, start, avg_time_per):
	steps_left = total_steps - step
	time_left = steps_left * avg_time_per / 60
	str1 = "{}/{} (epoch {}), train_loss: {:.5f}, ".format(step, total_steps, epoch, item["train_loss"])
	str2 = "lr: {:.6f}\n\ttime/{}: {:.3f}".format(item["lr"], print_cycle, end - start)
	str3 = " time/step = {:.3f}  time left: {:.2f}m g_step: {}".format(avg_time_per, time_left, item["g_step"])
	print(str1 + str2 + str3)
	assert step == item["g_step"], "Steps to not equal {} != {}".format(step, item["g_step"])


def pretty_print_confusion(confusion):
	print("Confusion:")
	for key in confusion.keys():
		print("\t{}: {:.5f}".format(key, confusion[key]))
	print("\n")


@format_float(precision=3)
def to_gb(num_bytes):
	return num_bytes / math.pow(2, 30)


@format_float(precision=3)
def to_mb(num_bytes):
	return num_bytes / math.pow(2, 20)


def save_model(args, saver, sess, step, dump=True, verbose=True):
	checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
	saver.save(sess, checkpoint_path, global_step=step)
	if dump:
		dump_args(args)
	if verbose:
		print("model saved to {}".format(checkpoint_path))


class Confusion(object):
	def __init__(self, sess, model, feed):
		self.sess = sess
		self.model = model
		self.feed = feed

	def __enter__(self):
		return self.sess.run(self.model.confusion, self.feed)

	def __exit__(self, error_type, value, trace):
		if value:
			print("Confusion Error: {}\n{}\n".format(error_type, value))
			traceback.print_tb(trace)
			exit(1)


class NormalTrain(object):
	def __init__(self, sess, model, feed):
		self.sess = sess
		self.feed = feed
		self.args = [model.cost, model.final_state, model.train_op]

	def __enter__(self):
		cost, state, _ = self.sess.run(self.args, self.feed)
		return state

	def __exit__(self, error_type, value, trace):
		if value:
			print("NormalTrain Error: {}\n{}\n{}".format(error_type, value, trace))
			exit(1)


class PrintTrain(object):
	def __init__(self, sess, model, summaries, feed):
		self.sess = sess
		self.feed = feed
		self.args = [summaries, model.loss, model.final_state, model.train_op, model.lr, model.global_step]

	def __enter__(self):
		summary, loss, state, _, lr, g_step = self.sess.run(self.args, feed_dict=self.feed)
		return {"summary": summary, "train_loss": loss, "state": state, "lr": lr, "g_step": g_step}

	def __exit__(self, error_type, value, trace):
		if value:
			print("PrintTrain Error: {}\n{}\n".format(error_type, value))
			traceback.print_tb(trace)
			exit(1)


def hidden_size(num_in, num_out):
	upper = num_out if num_out > num_in else num_in
	lower = num_in if num_in < num_out else num_out
	upper = upper if 2 * num_in < upper else 2 * num_in
	mid = (2 / 3) * num_in + num_out
	print("\n upper: {:.1f}   mid: {:.1f}	lower: {:.1f}:".format(upper, mid, lower))
	if upper > mid > lower:
		return int(mid)
	return int(upper + lower) // 2


def check_confusion(raw, theirs):
	assert raw["fn"] == theirs[1, 0], "False negatives don't match"
	assert raw["fp"] == theirs[0, 1], "False positives don't match"
	assert raw["tn"] == theirs[0, 0], "True negatives don't match"
	assert raw["tp"] == theirs[1, 1], "True positives don't match"
	assert raw["sum"] == np.sum(np.ndarray.flatten(np.array(theirs))), "Sums don't match"


def bucket_by_length(words):
	"""Bucket a list of words based on their lengths"""
	num_buckets = list(set([len(x) for x in words]))
	buckets = dict()
	for x in num_buckets:
		buckets[x] = list()
	for word in words:
		buckets[len(word)].append(word)
	return buckets

def train(args):
	one_mil = 1000000

	todo = 1 * one_mil



	data_loader = TextLoader(args.data_dir, args.save_dir,
							 args.batch_size, args.seq_length, todo=todo,
							 labeler_fn=labeler, is_training=True)

	args.vocab_size = data_loader.vocab_size
	args.batch_size = data_loader.batch_size
	args.label_ratio = data_loader.ratio
	args.num_classes = data_loader.num_classes

	print("Vocab size: ", args.vocab_size)
	print("Num classes: ", args.num_classes)

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

	# args.rnn_size = abs( (data_loader.vocab_size + args.seq_length) // 2)
	# print("Changed rnn size to be the avg of the in and out layers: ", args.rnn_size)
	# print("In layer: {}  Out layer: {}".format(args.seq_length, data_loader.vocab_size))

	# args.rnn_size = hidden_size(args.seq_length, data_loader.vocab_size)

	print("Changed rnn size to:", args.rnn_size)

	dump_data(data_loader, args)

	print_cycle = args.print_cycle
	total_steps = args.num_epochs * data_loader.num_batches

	print("Building model")
	model = Model(args, data_loader.num_batches)

	print("Model built")

	# used if you want a lot of logging
	sess_config = tf.ConfigProto()

	# used to watch gpu memory thats actually used
	sess_config.gpu_options.allow_growth = True

	# used to show where things are being placed
	sess_config.log_device_placement = False

	jit_level = 0
	jit_level = tf.OptimizerOptions.ON_1

	sess_config.graph_options.optimizer_options.global_jit_level = jit_level

	run_options = tf.RunOptions()
	run_options.trace_level = tf.RunOptions.FULL_TRACE

	run_meta = tf.RunMetadata()

	# set up some data capture lists
	global_start = time.time()

	args.data = dict()

	args.data["losses"] = list()
	args.data["avg_time_per_step"] = list()
	args.data["logged_time"] = list()

	args.num_params = int(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

	# refresh dumped data since some models
	# change the values of args
	dump_data(data_loader, model.args)
	dump_args(args)

	print("\nModel has {:,} trainable params".format(args.num_params))
	print("Data has {:,} individual characters\n".format(data_loader.num_chars))

	with tf.Session(config=sess_config) as sess:
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
		print("Have {} epochs and {} batches per epoch"
			  .format(args.num_epochs, data_loader.num_batches))
		total_time = 0.0
		# run_meta = tf.RunMetadata()

		data_loader.reset_batch_pointer()

		trace = None

		print("Initializing local variables")
		sess.run(tf.local_variables_initializer())

		for epoch in range(args.num_epochs):
			print("Resetting batch pointer for epoch: ", epoch)
			data_loader.reset_batch_pointer()
			# state = sess.run(model.initial_state)

			start = time.time()

			# This will get us our initial cell state of zero
			cell_state = None
			step = 0
			x, y = data_loader.next_batch()
			feed = {model.input_data: x, model.targets: y}
			with NormalTrain(sess, model, feed) as state:
				cell_state = state

			# after we have our "primed" network, we pass the state in
			# from the prev step to override the zero state of the model
			for batch in range(1, data_loader.num_batches):
				step = epoch * data_loader.num_batches + batch

				x, y = data_loader.next_batch()

				feed = {model.input_data: x, model.targets: y, model.cell_state: cell_state}
				# feed = {model.input_data: x, model.targets: y}


				last_batch = batch == data_loader.num_batches - 1
				last_epoch = epoch == args.num_epochs - 1

				# last_in_epoch = (step + 1) % data_loader.batch_size == data_loader.batch_size - 1
				s = epoch * data_loader.num_batches
				s = step - s
				last_in_epoch = s == data_loader.num_batches - 1

				# if printing
				if last_in_epoch or step == data_loader.num_batches  - 1 or step % print_cycle == 0 and step > 0 or (last_batch and last_epoch):
					do_print = False
					print("\n\n")
					with PrintTrain(sess, model, summaries, feed) as item:
						writer.add_summary(item["summary"], step)
						end = time.time()

						cell_state = item["state"]

						total_time += end - start
						avg_time_per = round(total_time / step if step > 0 else step + 1, 2)

						their_confusion = sess.run(model.their_confusion, feed)
						print(their_confusion)

							# only print weights data if we are using them
						if args.use_weights:
							print("Scale factor: ", sess.run(model.loss_scale_factors, feed))
							weights = sess.run(model.loss_weights, feed)
							sum_weights = [np.sum(x) for x in weights]
							print("Weights:", sum_weights)

						with Confusion(sess, model, feed) as confusion_matrix:
							pretty_print(item, step, total_steps, epoch, print_cycle, end, start, avg_time_per)

						start = time.time()

						global_diff = time.time() - global_start
						args.data["losses"].append(float(item["train_loss"]))
						args.data["logged_time"].append(int(global_diff))
						args.data["avg_time_per_step"].append(float(avg_time_per))
						args.data["last_recorded_loss"] = {
							"time": int(time.time() - global_start),
							"loss": float(item["train_loss"])
						}
						args.data["total_train_time"] = {
							"steps": int(total_steps),
							"time": int(time.time() - global_start)
						}

				else:  # else normal training
					with NormalTrain(sess, model, feed) as state:
						cell_state = state

				if last_in_epoch or step % args.save_every == 0 or (last_batch and last_epoch):
					# save for the last result

					# if trace:
					#	with open(os.path.join(args.save_dir, "step_" + str(step) + ".ctf.json"), "w") as t_file:
					#		t_file.write(trace.generate_chrome_trace_format())

					save_model(args, saver, sess, step, dump=False, verbose=False)
				if last_batch or last_epoch:
					dump_args(args)
				# increment the model step
				# model.inc_step()
				# sess.run(model.inc_step)
	dump_args(args)


if __name__ == '__main__':
	main()

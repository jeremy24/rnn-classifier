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

from utils import TextLoader
from model import Model
from data_blob import Prepositions
import numpy as np
import objgraph

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
	str2 = "lr: {:.6f}  time/{}: {:.3f}".format(item["lr"], print_cycle, end - start)
	str3 = " time/step = {:.3f}  time left: {:.2f}m g_step: {}".format(avg_time_per, time_left, item["g_step"])
	print(str1 + str2 + str3)


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


def save_model(args, saver, sess, step):
	checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
	saver.save(sess, checkpoint_path, global_step=step)
	dump_args(args)
	print("model saved to {}".format(checkpoint_path))


class Confusion(object):
	def __init__(self, sess, model, feed):
		self.sess = sess
		self.model = model
		self.feed = feed

	def __enter__(self):
		return self.sess.run(self.model.confusion, self.feed)

	def __exit__(self, type, value, trace):
		if value:
			print("Confusion Error: {}\n{}\n".format(type, value))
			traceback.print_tb(trace)
			exit(1)


class NormalTrain(object):
	def __init__(self, sess, model, feed):
		self.sess = sess
		self.feed = feed
		self.args = [model.cost, model.final_state, model.train_op]

	def __enter__(self):
		cost, state, _ = self.sess.run(self.args, self.feed)
		return None

	def __exit__(self, type, value, trace):
		if value:
			print("NormalTrain Error: {}\n{}\n{}".format(type, value, trace))
			exit(1)


class PrintTrain(object):
	def __init__(self, sess, model, summaries, feed):
		self.sess = sess
		self.feed = feed
		self.args = [summaries, model.loss, model.final_state, model.train_op, model.lr_decay, model.global_step]

	def __enter__(self):
		summary, loss, state, _, lr, g_step = self.sess.run(self.args, feed_dict=self.feed)
		return {"summary": summary, "train_loss": loss, "state": state, "lr": lr, "g_step": g_step}

	def __exit__(self, type, value, trace):
		if value:
			print("PrintTrain Error: {}\n{}\n".format(type, value))
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


def bucket_by_length(words):
	"""Bucket a list of words based on their lengths"""
	num_buckets = list(set([len(x) for x in words]))
	buckets = dict()
	for x in num_buckets:
		buckets[x] = list()
	for word in words:
		buckets[len(word)].append(word)
	return buckets


# generate labels
# first we bucket the words based on size and then mash them
# all into one regex for speed
# we have to do this in order to know the right number of replacement
# chars to sub into the string
def labeler(seq, words_to_use=5):
	"""Generate labels for a given sequence"""
	print("\nLabeler:")
	print("\tSeq length: {:,} ".format(len(seq)))
	a = seq
	# words = Prepositions().len_between(0, 6)
	# words = ["the", "of", "and", "in", "to", "a", "with", "for", "is"]
	words = list()

	# the word list is the top 10 most
	# common words in the sequence
	print("\tSplitting")
	b = seq.split(" ")
	wc = dict()
	for x in b:
		if x not in wc:
			wc[x] = 0
		wc[x] += 1

	print("\nWords being used:")
	for w in sorted(wc, key=wc.get, reverse=True):
		if len(words) == words_to_use:
			break
		print("\t{}:  {:,}".format(w, wc[w]))
		words.append(w)

	print("\nGenerating labels based on {} words".format(len(words)))

	# expressions = list()
	replace_char = chr(1)
	ret = np.zeros(len(a), dtype=np.uint8)

	# expressions = [make_exp(x) for x in words]

	expressions = [r"a", r"e", r"i", r"o", r"u"]
	words = ["a", "e", "i", "o", "u"]
	# each replace string is [ XXXX ] where X is the replace_char
	i = 0
	for word, exp in zip(words, expressions):
		gc.collect()
		replace_string = replace_char * len(word) # (" " + replace_char * len(word) + " ")
		a = re.sub(exp, replace_string, a)
		print("\t{:02d}: Done with: {}".format(i, word))
		i += 1

	print("\n\tDone with all replacements")

	for i in range(len(a)):
		ret[i] = a[i] == replace_char

	assert len(ret) == len(seq)
	return ret


def train(args):
	one_mil = 1000000

	todo = 1 * one_mil

	data_loader = TextLoader(args.data_dir, args.save_dir,
							 args.batch_size, args.seq_length, todo=todo,
							 labeler_fn=labeler)

	args.vocab_size = data_loader.vocab_size
	args.batch_size = data_loader.batch_size
	args.label_ratio = data_loader.ratio

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

		print("Total size of batch data: ", to_gb(data_loader.batches.nbytes), "GB")

		trace = None

		print("Initializing local variables")
		sess.run(tf.local_variables_initializer())

		for epoch in range(args.num_epochs):
			print("Resetting batch pointer for epoch: ", epoch)
			data_loader.reset_batch_pointer()
			# state = sess.run(model.initial_state)

			start = time.time()

			for batch in range(data_loader.num_batches):
				step = epoch * data_loader.num_batches + batch

				x, y = data_loader.next_batch()
				feed = {model.input_data: x, model.targets: y}

				last_batch = batch == data_loader.num_batches - 1
				last_epoch = epoch == args.num_epochs - 1

				# if printing
				if step % print_cycle == 0 and step > 0 or (last_batch and last_epoch):

					with PrintTrain(sess, model, summaries, feed) as item:
						writer.add_summary(item["summary"], step)
						end = time.time()

						total_time += end - start
						avg_time_per = round(total_time / step if step > 0 else step + 1, 2)

						print("False Negatives: ", sess.run(model.false_negatives, feed))
						print("Abs diff: ", sess.run(model.absolute_prediction_diff, feed))
						print("Scale factor: ", sess.run(model.loss_scale_factors, feed))
						print("Loss weights: ", sess.run(model.loss_weights, feed))

						with Confusion(sess, model, feed) as confusion_matrix:
							pretty_print(item, step, total_steps, epoch, print_cycle, end, start, avg_time_per)
							print(confusion_matrix)
							# pretty_print_confusion(confusion_matrix)
							# print("\nReferences:")
							# for thing, count in objgraph.most_common_types(limit=10):
							# 	print("\t{}: {:,}".format(thing, count))
							# objgraph.show_growth()

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
					with NormalTrain(sess, model, feed):
						pass

				if step % args.save_every == 0 or (last_batch and last_epoch):
					# save for the last result

					# if trace:
					#	with open(os.path.join(args.save_dir, "step_" + str(step) + ".ctf.json"), "w") as t_file:
					#		t_file.write(trace.generate_chrome_trace_format())

					save_model(args, saver, sess, step)

				# increment the model step
				# model.inc_step()
				# sess.run(model.inc_step)


if __name__ == '__main__':
	main()

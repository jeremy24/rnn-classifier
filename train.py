""" the model """
from __future__ import print_function

import argparse
import time
import os
import math
import json
import traceback
from six.moves import cPickle

from labeler import labeler, LabelTypes
from data_loader import TextLoader
from model import Model
from data_blob import Prepositions
import numpy as np

import tensorflow as tf

from decorators import *

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/extras/CUPTI/lib64/"


def dump_args(args):
	"""dump args to a file (json and cPickel)"""
	try:
		filename = os.path.join(args.save_dir, "hyper_params.json")
		backup = os.path.join(args.save_dir, "backup_hyper_params.json")
		with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
			cPickle.dump(args, f)
		with open(filename, "w") as fout:
			data = dict()
			vargs = vars(args)
			for key in vargs:
				data[key] = vargs[key]
			fout.write(json.dumps(data, sort_keys=True, indent=4, separators=(",", ":")))
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
	parser.add_argument('--max_gradient', type=float, default=5.,
						help='clip gradients at this value')
	parser.add_argument('--learning_rate', type=float, default=0.002,
						help='learning rate')
	parser.add_argument('--decay_rate', type=float, default=0.97,
						help='decay rate for rmsprop')
	parser.add_argument('--embedding_size', type=int, default=200,
						help='Size of the vocabulary embedding')
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
	# dump_args(args)

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


def pretty_print(item, step, total_steps, epoch, print_cycle, end, start, avg_time_per, x, y):
	steps_left = total_steps - step
	time_left = steps_left * avg_time_per / 60

	yes_labels = np.sum(np.array(y).flatten())
	ratio = yes_labels / len(np.array(x).flatten()) * 100.0
	print(item["confusion"])
	str1 = "{}/{} (epoch {}), train_loss: {:.5f}, ".format(step, total_steps, epoch, item["train_loss"])
	str2 = "lr: {:.6f}  label ratio: {:.5f}%\n\ttime/{}: {:.3f}".format(item["lr"], ratio, print_cycle, end - start)
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
		self.args = [model.cost, model.final_state, model.train_op, model.loss]

	def __enter__(self):
		cost, state, _, loss = self.sess.run(self.args, self.feed)
		return state, loss

	def __exit__(self, error_type, value, trace):
		if value:
			print("NormalTrain Error: {}\n{}\n{}".format(error_type, value, trace))
			exit(1)


class PrintTrain(object):
	def __init__(self, sess, model, summaries, feed):
		self.sess = sess
		self.feed = feed
		self.args = [summaries, model.loss, model.final_state,
					 model.train_op, model.lr, model.global_step, model.their_confusion]

	def __enter__(self):
		summary, loss, state, _, lr, g_step, confusion = self.sess.run(self.args, feed_dict=self.feed)
		return {"summary": summary, "train_loss": loss,
				"state": state, "lr": lr, "g_step": g_step, "confusion": confusion}

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


class dump_into_namespace:
	def __init__(self, env, *args):
		self.vars = dict([(x, env[x]) for v in args for x in env if v is env[x]])

	def __getattr__(self, name):
		print(self.vars)
		return self.vars[name]


def init_globals(sess):
	print("Saving global variables")
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver(tf.global_variables())
	return saver


def get_sess_config():
	# used if you want a lot of logging
	sess_config = tf.ConfigProto()

	# used to watch gpu memory thats actually used
	sess_config.gpu_options.allow_growth = True

	# used to show where things are being placed
	sess_config.log_device_placement = False

	jit_level = tf.OptimizerOptions.ON_1

	sess_config.graph_options.optimizer_options.global_jit_level = jit_level
	return sess_config


def copy_data_info(args, data_loader):
	args.vocab_size = data_loader.vocab_size
	args.batch_size = data_loader.batch_size
	args.label_ratio = data_loader.ratio
	args.num_classes = data_loader.num_classes
	args.num_batches = data_loader.num_batches
	args.num_chars = data_loader.num_chars
	return args

def get_flags(num_batches, num_epochs, epoch, batch, step):
	last_batch = batch == num_batches - 1
	last_epoch = epoch == num_epochs - 1
	s = step - (epoch * num_batches)
	last_in_epoch = s == num_batches - 1
	return last_batch, last_epoch, last_in_epoch


def do_init(args, data_loader):
	if args.init_from is not None:
		print("Initing from saved model")
		checkpoint = tf.train.get_checkpoint_state(args.init_from)
		assert checkpoint, "No checkpoint found"
		assert checkpoint.model_checkpoint_path, "No model path found in checkpoint"

		# merge the saved args with the current args
		# favor the saved versions
		with open(os.path.join(args.init_from, "hyper_params.json")) as saved_args:
			saved = json.load(saved_args)
			vargs = vars(args)
			for key in vargs:
				if key not in saved:
					saved[key] = vargs[key]
				saved[key] = vargs[key] if vargs[key] else saved[key]
			args = argparse.Namespace(**saved)
	else:
		# if we inited from a saved model this should already be loaded into the saved params
		print("\nSetting values from data_loader\n")
		args = copy_data_info(args, data_loader)
		checkpoint = None
	return args, checkpoint


def train(args):
	one_mil = 1000000

	todo = 1 * one_mil

	if not args.init_from:
		args.init_from = None

	data_loader = TextLoader(args.data_dir, args.save_dir,
							 args.batch_size, args.seq_length, todo=todo,
							 labeler_fn=labeler, is_training=True, max_word_length=None,
							 using_real_data=True)
	exit(1)
	# check compatibility if training is continued from previously saved model
	args, checkpoint = do_init(args, data_loader)

	print("Vocab size: ", args.vocab_size)
	print("Num classes: ", args.num_classes)
	print("Label Ratio: ", args.label_ratio)
	print("Changed rnn size to:", args.rnn_size)
	
	# print("\nDumping args data")
	# dump_data(data_loader, args)

	# setup printing cycle
	print_cycle = args.print_cycle
	total_steps = args.num_epochs * len(data_loader.train_batches)

	print("Building model")
	model = Model(args, len(data_loader.train_batches))
	print("Model built")

	sess_config = get_sess_config()

	run_options = tf.RunOptions()
	run_options.trace_level = tf.RunOptions.FULL_TRACE

	run_meta = tf.RunMetadata()

	# set up some data capture lists
	global_start = time.time()

	# setup stuff for more logging 
	# so we can save it to JSON
	args.data = dict()
	args.data["losses"] = list()
	args.data["avg_time_per_step"] = list()
	args.data["logged_time"] = list()
	
	# the number of trainable params in the model
	args.num_params = int(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

	# refresh dumped data since some models
	# change the values of args
	# dump_data(data_loader, model.args)
	# dump_args(args)

	print("\nModel has {:,} trainable params".format(args.num_params))
	print("Data has {:,} individual characters\n".format(data_loader.num_chars))

	# we are training, so set num batches accordingly
	data_loader.num_batches = len(data_loader.train_batches)

	with tf.Session(config=sess_config) as sess:
		# instrument for tensorboard
		summaries = tf.summary.merge_all()
		writer = tf.summary.FileWriter(
			os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))
		writer.add_graph(sess.graph)

		# get the graph saver
		saver = init_globals(sess)
		
		# restore model
		if args.init_from is not None and checkpoint:
			saver.restore(sess, checkpoint.model_checkpoint_path)

		print("Starting...")
		print("Have {} epochs and {} batches per epoch"
			  .format(args.num_epochs, len(data_loader.train_batches)))

		total_time = 0.0

		print("Initializing local variables")
		sess.run(tf.local_variables_initializer())

		print("\nStarting training loop, have {} steps until completion\n".format(total_steps))

		# start loss as infinity
		lowest_epoch_loss = float("inf")
		highest_true_positives = float("-inf")
		lowest_false_positives = float("inf")
		patience = 2

		# declare here so final save can access it
		step = 0
		args.epoch_stopped_on = -1

		for epoch in range(args.num_epochs):
			print("\nEpoch ", epoch)
			print("\tResetting batch pointer...")
			data_loader.reset_batch_pointer(quiet=True)
			# clear the epoch loss list
			epoch_loss = list()

			true_positives = list()
			false_positives = list()

			if epoch > patience:
				print("Patience is larger than epoch, breaking at Epoch {}".format(epoch))
				args.epoch_stopped_on = epoch
				break
			print("\tPatience: {:,}\n\tLast loss: {:.5f}".format(patience, lowest_epoch_loss))

			print_cycle_time = time.time()

			# This will get us our initial cell state of zero
			# since the model starts with a zero state by default
			# cell_state = None
			x, y = data_loader.next_train_batch()
			feed = {model.input_data: x, model.targets: y}
			with NormalTrain(sess, model, feed) as (state, loss):
				cell_state = state
				epoch_loss.append(loss)

			# after we have our "primed" network, we pass the state in
			# from the prev step to override the zero state of the model
			for batch in range(1, len(data_loader.train_batches)):
				step = epoch * data_loader.num_batches + batch

				# get data, make feed dict and pass in prev cell state
				x, y = data_loader.next_train_batch()
				feed = {model.input_data: x, model.targets: y, model.cell_state: cell_state}

				# get some of our flags to do things on certain steps
				last_batch, last_epoch, last_in_epoch = get_flags(data_loader.num_batches, args.num_epochs, epoch, batch, step)

				# if printing
				if last_in_epoch or step == data_loader.num_batches - 1 or step % print_cycle == 0 and step > 0 or (
					last_batch and last_epoch):
					print("\n\n")
					with PrintTrain(sess, model, summaries, feed) as item:
						writer.add_summary(item["summary"], step)
						print_cycle_end = time.time()

						cell_state = item["state"]

						# get some times
						total_time += print_cycle_end - print_cycle_time
						avg_time_per = round(total_time / (step if step > 0 else step + 1), 2)

						# only print weights data if we are using them
						if args.use_weights:
							print("Scale factor: ", sess.run(model.loss_scale_factors, feed))
							weights = sess.run(model.loss_weights, feed)
							sum_weights = [np.sum(x) for x in weights]
							print("Weights:  TP: {:.3f}  TN: {:.3f}  FN: {:.3f}  FP: {:.3f}".format(*sum_weights))

						# with Confusion(sess, model, feed) as confusion_matrix:
						pretty_print(item, step, total_steps, epoch, print_cycle, print_cycle_end, print_cycle_time, avg_time_per, x, y)

						# reset the timer
						print_cycle_time = time.time()

						epoch_loss.append(item["train_loss"])

						global_diff = time.time() - global_start
						args.data["losses"].append(float(item["train_loss"]))
						args.data["logged_time"].append(int(global_diff))
						args.data["avg_time_per_step"].append(float(avg_time_per))
						args.data["last_recorded_loss"] = {
							"time": int(time.time() - global_start),
							"loss": float(item["train_loss"]),
							"step": int(step)
						}
						args.data["total_train_time"] = {
							"steps": int(total_steps),
							"time": int(time.time() - global_start)
						}

				else:  # else normal training
					with NormalTrain(sess, model, feed) as (state, loss):
						cell_state = state
						epoch_loss.append(loss)

				# save for the last result
				if last_in_epoch or step % args.save_every == 0 or (last_batch and last_epoch):
					save_model(args, saver, sess, step, dump=True, verbose=False)

				conf = sess.run(model.their_confusion, feed)
				if len(conf) == 2 and len(conf[1]) == 2:
					true_positives.append(conf[1][1])
					false_positives.append(conf[0][1])

			highest_true_positives = float("-inf")
			lowest_false_positives = float("inf")

			if np.median(true_positives) > highest_true_positives:
				highest_true_positives = np.median(true_positives)
				print("New highest true positives: ", highest_true_positives)

			if np.median(false_positives) < lowest_false_positives:
				lowest_false_positives = np.median(false_positives)
				print("New lowest false positives: ", lowest_false_positives)

			# if new epoch loss is 0.5% lower than lowest loss, extend patience
			this_epoch_loss = np.median(epoch_loss)
			if this_epoch_loss + (this_epoch_loss * .008) < lowest_epoch_loss:
				patience += 2
				lowest_epoch_loss = this_epoch_loss
				print("Added 2 to patience, new lowest loss is {:.5f}".format(lowest_epoch_loss))
			elif this_epoch_loss + (this_epoch_loss * .005) < lowest_epoch_loss:
				patience += 1
				lowest_epoch_loss = this_epoch_loss
				print("Added 1 to patience, new lowest loss is {:.5f}".format(lowest_epoch_loss))
			elif this_epoch_loss + (this_epoch_loss * .008) > lowest_epoch_loss:
				patience -= 2
				print("Removing 2 from patience")
			elif this_epoch_loss + (this_epoch_loss * .005) > lowest_epoch_loss:
						patience -= 1
						print("Removing 1 from patience")

		# save model after all batches are done
		print("\nTraining is done, saving model")
		save_model(args, saver, sess, step, dump=True, verbose=False)


if __name__ == '__main__':
	main()

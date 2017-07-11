from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math

import sklearn as sk
import tensorflow as tf

from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq as s2s

import numpy as np

from decorators import *

"""Build a RNN model """


class Model(object):
	""" The RNN model """

	def add_dropout(self, cells, in_prob=1.0, out_prob=1.0):
		""" add dropout wrappers to cell[s], recursive """
		if type(cells) is not list:
			return rnn.DropoutWrapper(cells, input_keep_prob=in_prob,
									  output_keep_prob=out_prob)
		else:
			ret = [self.add_dropout(cell, in_prob, out_prob) for cell in cells]
			return ret

	def build_outputs(self, inputs):
		"""
		:param inputs:
		:return: outputs, last_state
		"""
		with tf.name_scope("outputs"):

			print("\nBuilding decoder helper")

			if not self.is_training:
				# if not training we still setup the training helper so the data i
				# is passed through, but we manually decode it first
				print("Using the inference helper")
				seq_lens = tf.fill([self.args.batch_size], self.args.seq_length)
				embedded_inputs = tf.nn.embedding_lookup(self.embedding,
														 tf.to_int32(self.input_data))
				decoder_helper = s2s.TrainingHelper(embedded_inputs, seq_lens)
			else:
				print("Using the training helper:")
				seq_lens = tf.fill([self.args.batch_size], self.args.seq_length)
				print("\tseq_lens: ", seq_lens.shape)
				print("\tinputs shape: ", inputs.shape)
				decoder_helper = s2s.TrainingHelper(inputs, seq_lens)

			# the meat
			decoder = s2s.BasicDecoder(self.cell, decoder_helper, self.initial_state)

			# what we want
			decoder_output, last_state, output_len = s2s.dynamic_decode(decoder)
			outputs = decoder_output.rnn_output

			print("Decoder outputs converted to floats")
			return tf.to_float(outputs), last_state

	def hang_gpu_variables(self):

		args = self.args
		print("\nHanging GPU variables")
		batch_shape = [args.batch_size, args.seq_length]

		print("\tbatch_shape: ", batch_shape)

		self.step = tf.Variable(0, dtype=self.gpu_type, trainable=False, name="step")
		self.input_data = tf.placeholder(self.gpu_type, shape=batch_shape, name="input_data")
		self.targets = tf.placeholder(self.gpu_type, shape=batch_shape, name="targets")

	def build_three_layers(self):
		"""
		Build the cells for a three layer network
		:param use_highway:
		:return: a list of LSTM cells
		"""
		# only working number of layers right now
		print("\nHave three layers, making sandwich...")
		cells = []

		print("\tStarting rnn size: ", self.args.rnn_size)
		print("\tStarting seq_length: ", self.args.seq_length)

		outer_size = self.args.rnn_size
		middle_size = outer_size // 2

		if True:
			self.args.rnn_size = outer_size
			print("\tChanged RNN size to: ", self.args.rnn_size)

		print("\tOuter size: {}  Middle size: {}".format(outer_size,
														 middle_size))

		# set up intersection stuff
		highway = tf.contrib.rnn.HighwayWrapper

		# project it onto the middle
		first = self.cell_fn(outer_size)
		middle = self.cell_fn(outer_size)
		last = self.cell_fn(outer_size)

		# dropout on first and last
		first = self.add_dropout(first, self.args.input_keep_prob, self.args.output_keep_prob)
		middle = self.add_dropout(middle, self.args.input_keep_prob, self.args.output_keep_prob)
		last = self.add_dropout(last, self.args.input_keep_prob, self.args.output_keep_prob)

		cells.append(first)
		cells.append(middle)
		cells.append(last)
		return cells

	def build_one_cell(self, dropout=True):
		c = self.cell_fn(self.args.rnn_size)
		c = self.add_dropout(c, self.args.input_keep_prob, self.args.output_keep_prob)
		return c

	def build_one_layer(self):
		# size = (self.args.seq_length + self.args.vocab_size) // 2
		# print("\nsize changed to {}\n".format(size))
		# self.args.rnn_size = size
		cell = self.cell_fn(self.args.rnn_size)
		cell = self.add_dropout(cell, self.args.input_keep_prob, self.args.output_keep_prob)
		return [cell]

	def build_cells(self):
		"""
		Build some RNN cells
		:return: a list of cells
		"""
		ret = []

		print("Building {} layers".format(self.args.num_layers))

		if self.args.num_layers == 1:
			ret = self.build_one_layer()
		# only working number of layers right now
		elif self.args.num_layers == 3:
			ret = self.build_three_layers()
		else:
			for x in range(self.args.num_layers):
				ret.append(self.build_one_cell())
			print("Do not have a routine to make {} layers, using default"
				  .format(self.args.num_layers))
		print("Done building cells")
		return ret

	def zero_states(self):
		pass

	def __init__(self, args, num_batches=None, training=True):
		""" init """

		if not training:
			print("\nNot training:")
			print("\tPrev batch size: ", args.batch_size)
			print("\tPrex seq length: ", args.seq_length)
		# we now leave setting these args up the to calling prgm
		# args.batch_size = 1
		# args.seq_length = 1

		self.args = args
		self.args.orig_batch_size = self.args.batch_size

		# the type of all variables through the system.
		# must be a float
		self.gpu_type = tf.float32
		self.is_training = training

		self.seq_length = self.args.seq_length

		# for the confusion matrix stuff
		self._confusion = None
		self._predictions = None
		self._recall_update = None
		self._recall = None
		self._accuracy = None
		self._accuracy_update = None
		self._precision = None
		self._precision_update = None

		self._lr_decay = None
		self._loss = None
		self._cost = None


		self.cell_fn = rnn.LSTMCell

		print("\nSetting self.lr = {:.5}".format(args.learning_rate))

		# self.lr = tf.Variable(args.learning_rate, name="lr", dtype=tf.float32)
		self._lr = None

		print("Cell type is: ", args.model)
		print("Batch size is: ", args.batch_size)

		# self.cell_fn = rnn.GRUCell
		# self.cell_fn = rnn.IntersectionRNNCell
		# self.cell_fn = rnn.NASCell


		# with tf.name_scope("cells"):
		# 	forward_cells = self.build_cells()
		# 	backward_cells = self.build_cells()
		# 	self.cell = tf.nn.bidirectional_dynamic_rnn(cell_fw=forward_cells,
		# 												cell_bw=backward_cells,
		# 												dtype=self.gpu_type,
		# 												sequence_length=self.seq_length)




		print("Setting self.num_batches")
		self.num_batches = num_batches

		# all teh data for the epoch
		# all of these are pinned to the gpu
		# self.all_input_data = None
		# self.all_target_data = None
		self._global_step = None
		# self.inc_step = tf.assign_add(self.global_step, 1, use_locking=True, name="inc_global_step")
		self.input_data = None
		self.targets = None
		self.step = None

		print("Calling hang_gpu_variables")
		# assign values to the above variables
		self.hang_gpu_variables()

		# self.batch = tf.train.batch(self.all_input_data, self.args.batch_size, name="input_batch_queue")
		# print("\nBatch size from tf.batch: ", self.batch.shape)

		# this maps vectors of len vocab_size => vectors of size rnn_size
		with tf.name_scope("get_embedding"):
			embedding = tf.get_variable("embedding",
										[args.vocab_size, args.rnn_size], trainable=False)
			inputs = tf.nn.embedding_lookup(embedding, tf.to_int32(self.input_data))
			self.embedding = embedding
			# inputs => [batch_size, seq_length, embedding_size]
			# embedding size is decided by tensorflow

		# # dropout beta testing: double check which one should affect next line
		# if training and args.output_keep_prob:
		# 	inputs = tf.nn.dropout(inputs, args.output_keep_prob)
		# # processing inputs on cpu
		# # with tf.device("/cpu:0"):

		# with tf.name_scope("cells"):
		self.forward_cells = rnn.MultiRNNCell(self.build_cells(), state_is_tuple=True)
		self.backward_cells = rnn.MultiRNNCell(self.build_cells(), state_is_tuple=True)

		# lets the user have a valid value to zero the cells out
		self.cell_zero_state = (self.forward_cells.zero_state(self.args.batch_size, self.gpu_type),
							  self.backward_cells.zero_state(self.args.batch_size, self.gpu_type))


		self.cell_state = (self.forward_cells.zero_state(self.args.batch_size, self.gpu_type),
							  self.backward_cells.zero_state(self.args.batch_size, self.gpu_type))

		print("Inputs: ", inputs)
		seq_lens = [self.seq_length for _ in range(self.args.batch_size)]
		output, self.final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.forward_cells,
														cell_bw=self.backward_cells,
														inputs=inputs,
														initial_state_fw=self.cell_state[0],
														initial_state_bw=self.cell_state[1],
														dtype=self.gpu_type,
														parallel_iterations=64,
														sequence_length=seq_lens)

		tf.summary.histogram("foward_state", self.cell_state[0])
		tf.summary.histogram("backward_state", self.cell_state[1])

		# if we are training, then make the last gotten state the new
		# initial state, else we are sampling or testing and we want
		# the starting state to be zero anyways
		# if self.is_training:
		# 	self.cell_state = self.last_state
		# else:
		# 	self.cell_state = (self.forward_cells.zero_state(self.args.batch_size, self.gpu_type),
		# 					  self.backward_cells.zero_state(self.args.batch_size, self.gpu_type))


		print(output)
		# exit(1)
		# average the outputs
		output = tf.divide(tf.add(output[0], output[1]), 2.0)

		# # This chunk is to do a regular multirnn setup
		# with tf.name_scope("Cells"):
		# 	print("Building cells of size: ", args.rnn_size)
		# 	forward_cells = self.build_cells()
		# 	print("\nCells:")
		# 	print("\tSquishing {} cells into one".format(len(forward_cells)))
		# 	for c in forward_cells:
		# 		print("\t", c)
		#
		# 	self.cell = rnn.MultiRNNCell(forward_cells, state_is_tuple=True)
		#
		# print("Setting self.initial_state based on batch size: ", self.args.batch_size)
		# self.initial_state = self.cell.zero_state(self.args.batch_size, self.gpu_type)
		# output, last_state = self.build_outputs(inputs)

		print("Getting logits")

		# the final layers
		# maps the outputs	to [ vocab_size ] probs
		# self.logits = tf.contrib.layers.fully_connected(output, args.vocab_size)
		self.logits = tf.layers.dense(inputs=output, units=self.args.num_classes)
		print("Logits shape: ", self.logits.shape)

		# both of these are for sampling
		with tf.name_scope("probabilities"):
			print("Getting probs")
			self.probs = tf.nn.softmax(self.logits, name="probs_softmax")

		# we assume predicting one at a time for now
		# TODO rewite to be able to predict N number of seqs at once
		if not self.is_training:
			with tf.name_scope("predict_index"):
				# set predict to the right series of operations
				self.predict = tf.squeeze(self.probs)

		# make into [ batch_size, seq_len, vocab_size ]
		# it should already be this size, but this forces tf to recognize
		# the shape
		logits_shape = [self.args.batch_size, self.args.seq_length, self.args.num_classes]
		self.logits = tf.reshape(self.logits, logits_shape)

		# with tf.name_scope("compute_loss"):
		# 	# loss_weights = tf.ones([self.args.batch_size, self.args.seq_length])
		# 	# self.loss = s2s.sequence_loss(split_logits, tf.to_int32(self.targets),
		# 	#							loss_weights, name="compute_loss")
		#
		# 	onehots = tf.one_hot(indices=tf.to_int32(self.targets),
		# 						 depth=self.args.vocab_size, dtype=tf.int32)
		# 	# self.loss = tf.losses.softmax_cross_entropy(onehot_labels=onehots, logits=self.logits)
		#
		# 	self.loss = tf.nn.weighted_cross_entropy_with_logits(targets=onehots, logits=self.logits, pos_weight=3.0)
		#
		# 	print("Vocab size:", self.args.vocab_size)
		# 	print("logits shape: ", self.logits.shape)
		# 	print("Targets shape: ", self.targets.shape)
		#
		# 	# self.confusion = self.compute_confusion(self.logits, tf.to_int32(self.targets))
		# # self.loss = -self.confusion["precision"]

		# self.final_state = last_state
		# print(self.lr)

		# self._lr = tf.Variable(self.args.learning_rate, name="lr", dtype=tf.float32, trainable=False)
		self.decay_rate = float(self.args.decay_rate)

		# make sure it's not one, else it will never decrease
		if self.args.learning_rate == 1:
			self.args.learning_rate = .9999

		self.global_step = tf.Variable(-1, name="global_step", trainable=False)

		self.min_learn_rate = .005

		self.lr_decay_fn = tf.train.exponential_decay(self.args.learning_rate,
												global_step=tf.assign_add(self.global_step, 1, use_locking=True, name="inc_global_step"),
												decay_steps=self.num_batches // 2,
												decay_rate=self.decay_rate,
												staircase=True, name="lr")

		# don't allow the learning rate to go below a certain minimum
		self.lr = self.args.learning_rate
		self.lr = tf.cond(tf.less(self.lr, self.min_learn_rate),
						  true_fn=lambda: self.min_learn_rate,
						  false_fn=lambda: self.lr_decay_fn)

		print("\nSetup learning rate decay:")
		print("\tlr: {}\n\tdecay every {} steps\n\tdecay rate: {}\n\tstaircase: {}"
			  .format(self.lr, self.num_batches // 2, self.decay_rate, True))

		with tf.name_scope("optimizer"):
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

		# put this here to make sure all internal optimizer variables get initialized
		with tf.name_scope("train"):
			self.train_op = self.optimizer.minimize(self.loss)


		# self.train_op = tf.contrib.layers.optimize_loss(
		# 	loss=self.loss,
		# 	global_step=self.global_step,
		# 	clip_gradients=self.args.grad_clip,
		# 	learning_rate_decay_fn=decay_fn,
		# 	name="optimize",
		# 	colocate_gradients_with_ops=None,
		# 	learning_rate=self.lr,
		# 	optimizer="Adagrad")

		print("\ntrainable_variables:")
		for var in tf.trainable_variables():
			print("\t", var)

		# values for tensorboard
		# some nice logging

		tf.summary.scalar("loss", self.loss)
		tf.summary.scalar("false_negatives", self.false_negatives)
		tf.summary.scalar("true_positives", self.true_positives)

		self.add_summaries(self.probs, "probability")

	@staticmethod
	def add_summaries(item, name):
		tf.summary.scalar("max_" + name, tf.reduce_max(item))
		tf.summary.scalar("min_" + name, tf.reduce_min(item))


	@ifnotdefined
	def onehot_labels(self):
		tf.one_hot(indices=tf.to_int32(self.targets),
				   depth=self.args.num_classes, dtype=self.targets.dtype)

	@define_scope
	def loss(self):
		# loss_weights = tf.ones([self.args.batch_size, self.args.seq_length])
		# self.loss = s2s.sequence_loss(split_logits, tf.to_int32(self.targets),
		#							loss_weights, name="compute_loss")
		print("\nSetting up loss")
		print("\tCalling confusion to hang its internals")
		confusion = self.confusion

		self.label_ratio = float(self.args.label_ratio)

		print("\tLabel Ratio: {:.3f}".format(self.label_ratio))
		onehots = tf.one_hot(indices=tf.to_int32(self.targets),
							 depth=self.args.num_classes, dtype=self.targets.dtype)

		# self._loss = tf.losses.softmax_cross_entropy(onehot_labels=onehots, logits=self.logits)
		print("\tOnehots shape: ", onehots.shape)
		print("\tLogits shape: ", self.logits.shape)
		# print("\tWeight shape: ", self.loss_weights.shape)
		assert onehots.shape == self.logits.shape, "Logits shape != labels shape"

		# if we don't use weights than all weights default to one
		if self.args.use_weights:
			print("\tWeighting the losses")
			# a = self.loss
			# self._loss = self.false_negative_loss_scale_factor
			self._loss = tf.losses.softmax_cross_entropy(onehot_labels=onehots,
														 logits=self.logits,
														 weights=tf.add_n(self.loss_weights))

			raw_loss = tf.losses.softmax_cross_entropy(onehot_labels=onehots,
														 logits=self.logits,
														 weights=tf.add_n(self.loss_weights),
														 reduction=tf.losses.Reduction.NONE)

			tf.summary.histogram("raw_loss", raw_loss)

		else:
			print("\tNot weighting the losses")
			self._loss = tf.losses.softmax_cross_entropy(onehot_labels=onehots,
														 logits=self.logits)

		print("Returning loss")
		# scale it up
		return self._loss + self.false_negative_loss_scale_factor  # tf.losses.get_total_loss()  # self._loss

	@define_scope
	def raw_matrix(self):
		ret = {"sum": 0, "fn": self.false_negatives, "tn": self.true_negatives,
			   "fp": self.false_positives, "tp": self.true_positives}
		ret["sum"] = ret["fn"] + ret["fp"] + ret["tn"] + ret["tp"]

		flat_preds = self.predictions  # tf.reshape(self.predictions, [-1])
		targets = tf.reshape(self.targets, [-1])

		print("Target shape", targets)
		print("Pred shape: ", flat_preds.shape)

		ret["fn_"] = tf.contrib.metrics.streaming_false_negatives(labels=targets, predictions=flat_preds)
		ret["fp_"] = tf.contrib.metrics.streaming_false_positives(labels=targets, predictions=flat_preds)
		ret["tp_"] = tf.contrib.metrics.streaming_true_positives(labels=targets, predictions=flat_preds)
		ret["fp_"] = self.false_positives

		return ret

	@property
	def int_targets(self):
		return tf.to_int32(self.targets)

	# @define_scope(summary=True)
	# def false_negatives(self):
	# 	inner = tf.multiply(self.int_targets, tf.to_int32(self.absolute_prediction_diff))
	# 	fn = tf.reduce_sum(inner)
	# 	return fn

	@define_scope(summary=True)
	def true_negatives(self):
		return (
			   self.args.batch_size * self.args.seq_length) - self.false_positives - self.true_positives - self.false_negatives

	@define_scope(summary=True)
	def false_positives(self):
		return tf.reduce_sum(
			tf.multiply(tf.to_int32(self.twod_predictions), tf.to_int32(self.absolute_prediction_diff)))

	# @define_scope(summary=True)
	# def true_positives(self):
	# 	return tf.reduce_sum(tf.multiply(tf.to_int32(self.twod_predictions), self.int_targets))

	@ifnotdefined
	def absolute_prediction_diff(self):
		return tf.losses.absolute_difference(labels=self.float_targets,
											 predictions=tf.to_float(self.twod_predictions),
											 reduction=tf.losses.Reduction.NONE)

	@ifnotdefined
	def float_targets(self):
		return tf.to_float(self.targets)

	@ifnotdefined
	def twod_predictions(self):
		predictions = tf.nn.softmax(self.logits)
		# assert tf.rank(predictions) > 2, "Can't get 2d predictions from tensor of rank < 3"
		return tf.argmax(predictions, 2)

	@ifnotdefined
	def loss_weights(self):
		targets = tf.to_float(self.targets)
		predictions = tf.to_float(self.twod_predictions)

		assert targets.shape == predictions.shape, "Targets shape != Predictions shape"

		preds = tf.cast(predictions, tf.bool)
		targs = tf.cast(targets, tf.bool)

		print("Ratio: ", self.label_ratio)

		# fn => guessed no was yes 0 and 1
		# fp => guessed yes was no 1 and 0
		# tp => guessed yes was yes 1 and 1
		# tn => guessed no was no  0 and 0

		# fn = pred is 0 and targ is yes
		# fp = pred is yes and targ is no

		false_negatives = tf.to_float(tf.logical_and(tf.logical_not(preds), targs))
		false_positives = tf.to_float(tf.logical_and(preds, tf.logical_not(targs)))

		true_negatives = tf.to_float(tf.logical_and(tf.logical_not(preds), tf.logical_not(targs)))
		true_positives = tf.to_float(tf.logical_and(preds, targs))

		# false_negatives = tf.to_float(tf.logical_and(negatives, falses))
		# false_positives = tf.to_float(tf.logical_and(positives, falses))
		#
		# true_positives = tf.to_float(tf.logical_and(positives, trues))
		# true_negatives = tf.to_float(tf.logical_and(negatives, trues))

		self.false_negatives = tf.to_int32(tf.reduce_sum(false_negatives))
		self.true_positives = tf.to_int32(tf.reduce_sum(true_positives))

		# make sure no zeros end up in the weights matrix
		scale = tf.cond(tf.greater(self.false_negative_loss_scale_factor, 0),
						true_fn=lambda: self.false_negative_loss_scale_factor,
						false_fn=lambda: 1.0)

		# scale = tf.add(scale, self.label_ratio)

		weighted_fp = tf.multiply(false_positives, 2.0)
		weighted_tp = tf.multiply(true_positives, 0.8)

		weighted_tn = tf.multiply(true_negatives, 1.0)
		weighted_fn = tf.multiply(false_negatives, scale)


		# the above weights will punish false negatives and reward true positives

		return [weighted_tp, weighted_tn, weighted_fn, weighted_fp]  # weights

	# @define_scope
	# def loss_scale_factor(self):
	# 	return tf.log(tf.to_float(tf.abs(self._fn)))

	# @define_scope(scope="print_scales")
	@property
	def loss_scale_factors(self):
		return {"fn": self.false_negative_loss_scale_factor + self.label_ratio}

	@staticmethod
	def loglog(item):
		return tf.log(tf.log(tf.to_float(item)))

	def fn_punish(self, number):
		# return number / 2.0
		return tf.log(number) #* ( 1.0 / self.args.label_ratio)

	@define_scope(scope="scale_factor")
	def false_negative_loss_scale_factor(self):
		"""
		If false negatives are zero then return one else
			return log(false negatives)
		This will hang a scalar summary to log the scale factor being used
		It should always return a real value and never +-inf or NaN
		"""
		if self.false_negatives is None:
			raise Exception("Cannot get loss_scale_factor, model.false_negatives is undefined ")

		as_float = tf.to_float(tf.abs(self.false_negatives))
		# value = tf.Variable(1.0, dtype=self.gpu_type, name="loss_scale_factor", trainable=False)
		# if its zero, return one (no punishment) since log(0) is undefined
		# this avoids NaN weights
		cond = tf.cond(tf.equal(as_float, 0.0),
					   true_fn=lambda: 1.0,
					   false_fn=lambda: self.fn_punish(as_float))
		tf.summary.scalar(name="loss_scale_factor", tensor=cond)
		return cond  # tf.assign(value, cond)

	@ifnotdefined
	def cost(self):
		cost = self.loss
		return cost

	@ifnotdefined
	def predictions(self):
		predictions = tf.reshape(tf.nn.softmax(self.logits), [-1, self.args.num_classes])
		predictions = tf.argmax(predictions, 1)
		return predictions

	@ifnotdefined
	def recall(self):
		targets = tf.reshape(self.targets, [-1])
		recall, _ = tf.metrics.recall(labels=targets, predictions=self.predictions)
		return recall

	@ifnotdefined
	def accuracy(self):
		targets = tf.reshape(self.targets, [-1])
		accuracy, _ = tf.metrics.accuracy(labels=targets, predictions=self.predictions)
		return accuracy

	@ifnotdefined
	def precision(self):
		targets = tf.reshape(self.targets, [-1])
		precision, _ = tf.metrics.precision(labels=targets, predictions=self.predictions)
		return tf.metrics.precision(labels=targets, predictions=self.predictions)

	@define_scope
	def their_confusion(self):
		return tf.confusion_matrix(labels=tf.reshape(self.targets, [-1]),
								   predictions=tf.reshape(self.predictions, [-1]))

	@ifnotdefined
	def confusion(self):
		return {"accuracy": self.accuracy, "precision": self.precision, "recall": self.recall}


	@define_scope
	def hardmax(self):
		return tf.squeeze(s2s.hardmax(self.logits))

	@define_scope
	def single_hardmax(self):
		return tf.argmax(self.hardmax)

	def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=0):
		# state = sess.run(self.cell.zero_state(1, tf.float32))
		print("got initial zero state")

		print("Sampling type: ", sampling_type)
		print("Primer: ", prime)

		for char in prime[:-1]:
			x = np.zeros((1, 1))
			x[0, 0] = vocab[char]
			feed = {self.input_data: x}  # , self.initial_state: state}
			[state] = sess.run([self.final_state], feed)

		print("Primed")

		def weighted_pick(weights):
			t = np.cumsum(weights)
			s = np.sum(weights)
			return int(np.searchsorted(t, np.random.rand(1) * s))

		ret = prime
		char = prime[-1]
		print("Starting")
		for n in range(num):
			x = np.zeros((1, 1))
			x[0, 0] = vocab[char]
			feed = {self.input_data: x}  # , self.initial_state: state}
			[probs, state] = sess.run([self.probs, self.final_state], feed)
			p = probs[0]

			if sampling_type == 0:
				sample = np.argmax(p)
			elif sampling_type == 2:
				if char == ' ':
					sample = weighted_pick(p)
				else:
					sample = np.argmax(p)
			else:  # sampling_type == 1 default:
				sample = weighted_pick(p)

			prediction = chars[sample]
			ret += prediction
			char = prediction
		return ret


	def _old_sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=0):
		state = sess.run(self.cell.zero_state(1, tf.float32))
		print("got initial zero state")

		print("Sampling type: ", sampling_type)
		print("Primer: ", prime)

		for char in prime[:-1]:
			x = np.zeros((1, 1))
			x[0, 0] = vocab[char]
			feed = {self.input_data: x, self.initial_state: state}
			[state] = sess.run([self.final_state], feed)

		print("Primed")

		def weighted_pick(weights):
			t = np.cumsum(weights)
			s = np.sum(weights)
			return int(np.searchsorted(t, np.random.rand(1) * s))

		ret = prime
		char = prime[-1]
		print("Starting")
		for n in range(num):
			x = np.zeros((1, 1))
			x[0, 0] = vocab[char]
			feed = {self.input_data: x, self.initial_state: state}
			[probs, state] = sess.run([self.probs, self.final_state], feed)
			p = probs[0]

			if sampling_type == 0:
				sample = np.argmax(p)
			elif sampling_type == 2:
				if char == ' ':
					sample = weighted_pick(p)
				else:
					sample = np.argmax(p)
			else:  # sampling_type == 1 default:
				sample = weighted_pick(p)

			prediction = chars[sample]
			ret += prediction
			char = prediction
		return ret

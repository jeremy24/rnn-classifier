from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

import sklearn as sk
import tensorflow as tf

from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq as s2s

import numpy as np

from decorators import *

"""Build a RNN model """


# Filter out INFO logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

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

	# this currently isn't being used right now
	# due to the overhaul to use a bi-directional RNN
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
				print("\tUsing the inference helper")
				seq_lens = tf.fill([self.args.batch_size], self.args.seq_length)
				embedded_inputs = tf.nn.embedding_lookup(self.embedding,
														 tf.to_int32(self.input_data))
				decoder_helper = s2s.TrainingHelper(embedded_inputs, seq_lens)
			else:
				print("\tUsing the training helper:")
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
		if dropout:
			c = self.add_dropout(c, self.args.input_keep_prob, self.args.output_keep_prob)
		return c

	def build_one_layer(self, size):
		cell = self.cell_fn(size)
		cell = self.add_dropout(cell, self.args.input_keep_prob, self.args.output_keep_prob)
		return [cell]

	def build_cells(self, size):
		"""
		Build some RNN cells
		:return: a list of cells
		"""
		ret = []
		print("\nBuilding cells:")
		print("\tMaking {} layers".format(self.args.num_layers))

		if self.args.num_layers == 1:
			ret = self.build_one_layer(size)
		# only working number of layers right now
		elif self.args.num_layers == 3:
			ret = self.build_three_layers()
		else:
			for x in range(self.args.num_layers):
				ret.append(self.build_one_cell())
			print("\tDo not have a routine to make {} layers, using default"
				  .format(self.args.num_layers))
		return ret

	def _variable_on_cpu(self, name, shape, initializer):
		"""Helper to create a Variable stored on CPU memory.
		Args:
		  name: name of the variable
		  shape: list of ints
		  initializer: initializer for Variable
		Returns:
		  Variable Tensor
		"""
		with tf.device('/cpu:0'):
			var = tf.get_variable(name, shape, initializer=initializer, dtype=self.gpu_type)
		return var

	def _variable_with_weight_decay(self, name, shape, stddev, wd):
		"""Helper to create an initialized Variable with weight decay.
		Note that the Variable is initialized with a truncated normal distribution.
		A weight decay is added only if one is specified.
		Args:
		  name: name of the variable
		  shape: list of ints
		  stddev: standard deviation of a truncated Gaussian
		  wd: add L2Loss weight decay multiplied by this float. If None, weight
			  decay is not added for this Variable.
		Returns:
		  Variable Tensor
		"""
		var = self._variable_on_cpu(
			name,
			shape,
			tf.truncated_normal_initializer(stddev=stddev, dtype=self.gpu_type))
		if wd is not None:
			weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
		return var

	def add_conv(self, conv_input, out_size, conv_stride, pool_stride, kernel_size, layer_number):
		print("\tLayer: ", layer_number)
		print("\t\tKernel: {}  Conv Stride: {} Pool Stride: {} Out Size: {}".format(kernel_size, conv_stride, pool_stride, out_size))
		cluster = tf.layers.conv1d(inputs=conv_input, filters=out_size,
								   kernel_size=[kernel_size],
								   strides=[conv_stride],
								   padding="SAME",
								   activation=lambda x: tf.maximum(0.0, x),
								   name="conv_" + layer_number)
		print("\t\tCluster: ", cluster.shape)
		pool = tf.layers.max_pooling1d(cluster, pool_size=3,
									   strides=[pool_stride],
									   padding="SAME",
									   name="pool_" + layer_number)
		print("\t\tPool: ", pool.shape)
		tf.summary.histogram("conv/conv_" + layer_number, cluster)
		tf.summary.histogram("conv/pool_" + layer_number, pool)
		return pool

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

		self.seq_length = int(self.args.seq_length)
		self.max_gradient = float(self.args.max_gradient)

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

		args.model = str(args.model).lower()

		print("\n")
		if args.model == "gru":
			print("Using GRU Cell")
			self.cell_fn = rnn.GRUCell
		elif args.model == "glstm":
			print("Using GLSTM Cell")
			self.cell_fn = rnn.GLSTMCell
		else:
			print("Using LSTM Cell")
			self.cell_fn = rnn.LSTMCell
		print("\n")

		print("\nSetting self.lr = {:.5}".format(args.learning_rate))

		# self.lr = tf.Variable(args.learning_rate, name="lr", dtype=tf.float32)
		self._lr = None

		print("Cell type is: ", args.model)
		print("Batch size is: ", args.batch_size)

		# self.cell_fn = rnn.GRUCell
		# self.cell_fn = rnn.IntersectionRNNCell
		# self.cell_fn = rnn.NASCell

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
										[args.vocab_size, args.embedding_size], trainable=False)
			inputs = tf.nn.embedding_lookup(embedding, tf.to_int32(self.input_data))
			self.embedding = embedding
		# inputs => [batch_size, seq_length, embedding_size]
		# embedding size is decided by tensorflow


		print("\nInputs: ", inputs.shape)

		# If using the conv in front of the rnn
		# conv_input = inputs
		#
		# print("\n\nClustering")
		# print("\tConv input: ", conv_input.shape)
		#
		# with tf.name_scope("swap_conv_dims"):
		# 	conv_input = tf.reshape(conv_input, [self.args.batch_size, self.seq_length, self.args.embedding_size])
		# 	conv_input = tf.transpose(conv_input, perm=[0, 2, 1])  # swap dims
		# 	# conv_input = tf.reshape(conv_input, shape=[self.args.batch_size, self.seq_length * self.args.rnn_size, 1])
		#
		# print("\tReshaped:   ", conv_input.shape)
		#
		# cluster_stride = 1
		# # cluster1 = tf.layers.conv1d(inputs=output, filters=self.args.seq_length,
		# # 							kernel_size=[7],
		# # 							strides=[cluster_stride],
		# # 							padding="SAME",
		# # 							activation=lambda x: tf.maximum(0.0, x),
		# # 							name="cluster1")
		# #
		# # print("\tRaw cluster1: ", cluster1.shape)
		# # print("\tCluster1:     ", cluster1.shape)
		# #
		# # # in => [ batch, height, width, channels ]
		# # # ksize => size of window for each dim
		# # # strides => stride of window for each dim
		# # # passing => SAME
		# #
		# #
		# # cluster_pool1 = tf.layers.max_pooling1d(cluster1, pool_size=3,
		# # 										strides=[2],
		# # 										padding="SAME",
		# # 										name="pool1")
		#
		# # print("\tClusterPool1: ", cluster_pool1.shape, "\n")
		#
		# # self, input, out_size, conv_stride, pool_stride, kernel_size, layer_number):
		#
		# print("\tSeq length: ", self.seq_length)
		#
		# # input, out_size, conv_stride, pool_stride, kernel_size, layer_number):
		#
		# cluster_pool1 = self.add_conv(conv_input=conv_input, out_size=self.seq_length, conv_stride=cluster_stride, pool_stride=2, kernel_size=7, layer_number="1")
		# print("\tLayer one:    ", cluster_pool1.shape)
		#
		# cluster_pool2 = self.add_conv(conv_input=cluster_pool1, out_size=self.seq_length, conv_stride=cluster_stride, pool_stride=2, kernel_size=5, layer_number="2")
		# print("\tLayer two:    ", cluster_pool2.shape)
		#
		# cluster_pool3 = self.add_conv(cluster_pool2, out_size=self.seq_length, conv_stride=cluster_stride, pool_stride=2, kernel_size=3, layer_number="3")
		# print("\tLayer three: ", cluster_pool3.shape)
		#
		# final_conv_out = cluster_pool3
		#
		# with tf.name_scope("swap_rnn_dims"):
		# 	cluster_output = tf.transpose(final_conv_out, perm=[0, 2, 1])
		# 	# cluster_output = tf.reshape(final_conv_out, shape=[self.args.batch_size, self.seq_length, -1])
		#
		#
		#
		# print("\tPool Output:  ", cluster_output.shape)
		# rnn_size = cluster_output.shape[2].value
		# rnn_input = cluster_output

		# # if using JUST the RNN
		rnn_size = self.args.embedding_size
		rnn_input = inputs


		print("\nRNN:")
		# print("\t", cluster_output)
		# print("\t", cluster_output.shape)
		# print("\t", dir(cluster_output.shape))

		print("\tSize:     ", rnn_size)
		# with tf.name_scope("cells"):
		self.forward_cells = rnn.MultiRNNCell(self.build_cells(size=rnn_size), state_is_tuple=True)
		self.backward_cells = rnn.MultiRNNCell(self.build_cells(size=rnn_size), state_is_tuple=True)

		# lets the user have a valid value to zero the cells out
		self.cell_zero_state = (self.forward_cells.zero_state(self.args.batch_size, self.gpu_type),
								self.backward_cells.zero_state(self.args.batch_size, self.gpu_type))

		self.cell_state = (self.forward_cells.zero_state(self.args.batch_size, self.gpu_type),
						   self.backward_cells.zero_state(self.args.batch_size, self.gpu_type))

		print("\tInput:    ", rnn_input.shape)
		seq_lens = [self.seq_length for _ in range(self.args.batch_size)]
		rnn_layer_out, self.final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.forward_cells,
																	   cell_bw=self.backward_cells,
																	   inputs=rnn_input,
																	   initial_state_fw=self.cell_state[0],
																	   initial_state_bw=self.cell_state[1],
																	   dtype=self.gpu_type,
																	   parallel_iterations=64,
																	   sequence_length=seq_lens)


		# average the outputs
		with tf.name_scope("avg_rnn_output"):
			print("\nPooling ")
			if "MODEL_USE_RNN_CONCAT" in os.environ and int(os.environ["MODEL_USE_RNN_CONCAT"]) == 1:
				print("\tUsing concatted outputs!")
				concatted = tf.concat(rnn_layer_out, 2)
				print("\tConcatted:  ", concatted.shape)
				rnn_output = concatted
			else:
				print("\tUsing averaged outputs")
				rnn_output = tf.divide(tf.add(rnn_layer_out[0], rnn_layer_out[1]), 2.0)


		# rnn_output = tf.cond(tf.equal(tf.mod(self.step, 150), 0),
		# 		true_fn=lambda: tf.add(raw_rnn_output, tf.random_normal(raw_rnn_output.shape, dtype=self.gpu_type)),
		# 		false_fn=lambda: raw_rnn_output)
		print("\tOutput:   ", rnn_output.shape)



		# exit(1)

		# rnn_output = tf.transpose(rnn_output, perm=[0, 2, 1])
		# rnn_pool = tf.layers.max_pooling1d(rnn_output, pool_size=3,
		# 							   strides=[2],
		# 							   padding="SAME",
		# 							   name="rnn_pool")
		#
		# print("\tRNN Pool: ", rnn_pool.shaoe)
		#
		# rnn_output = tf.transpose(rnn_pool, perm=[0, 2, 1])
		#
		# print("Pool Out: ", rnn_output.shape)


		tf.summary.histogram("cell/foward_state", self.cell_state[0])
		tf.summary.histogram("cell/backward_state", self.cell_state[1])

		# the final layers
		# maps the outputs	to [ vocab_size ] probs
		# self.logits = tf.contrib.layers.fully_connected(output, args.vocab_size)
		self.logits = tf.layers.dense(inputs=rnn_output, units=self.args.num_classes)

		# self.logits = tf.contrib.layers.fully_connected(inputs=rnn_output,
		# 												num_outputs=self.args.num_classes)

		# first logits shape => [ batch_size, seq_length, num_classes ]
		print("\nLogits shape: ", self.logits.shape)
		# self.args.rnn_size = self.logits.shape[1]

		# both of these are for sampling
		with tf.name_scope("probabilities"):
			print("\nGetting probs")
			self.probs = tf.nn.softmax(self.logits, name="probs_softmax")
			print("\tProbs: ", self.probs)

		# exit(1)

		# we assume predicting one at a time for now
		# TODO rewite to be able to predict N number of seqs at once
		if not self.is_training:
			with tf.name_scope("predict_index"):
				# set predict to the right series of operations
				self.predict = tf.squeeze(self.probs)

		# make into [ batch_size, seq_len, vocab_size ]
		# it should already be this size, but this forces tf to recognize
		# the shape
		# logits_shape = [self.args.batch_size, self.args.seq_length, self.args.num_classes]
		# self.logits = tf.reshape(self.logits, logits_shape)

		# self._lr = tf.Variable(self.args.learning_rate, name="lr", dtype=tf.float32, trainable=False)
		self.decay_rate = float(self.args.decay_rate)

		# make sure it's not one, else it will never decrease
		if self.args.learning_rate == 1:
			self.args.learning_rate = .9999

		self.global_step = tf.Variable(-1, name="global_step", trainable=False)

		self.min_learn_rate = .005

		self.lr_decay_fn = tf.train.exponential_decay(self.args.learning_rate,
													  global_step=tf.assign_add(self.global_step, 1, use_locking=True,
																				name="inc_global_step"),
													  decay_steps=self.num_batches,
													  decay_rate=self.decay_rate,
													  staircase=False, name="lr")

		# don't allow the learning rate to go below a certain minimum
		self.lr = self.args.learning_rate
		self.lr = tf.cond(tf.less(self.lr, self.min_learn_rate),
						  true_fn=lambda: self.min_learn_rate,
						  false_fn=lambda: self.lr_decay_fn)

		print("\nSetup learning rate decay:")
		print("\tlr: {}\n\tdecay every {} steps\n\tdecay rate: {}\n\tstaircase: {}"
			  .format(self.lr, self.num_batches // 2, self.decay_rate, True))

		# with tf.name_scope("optimizer"):
		# 	self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
		#
		# # put this here to make sure all internal optimizer variables get initialized
		# with tf.name_scope("train"):
		# 	self.train_op = self.optimizer.minimize(self.loss)


		tvars = tf.trainable_variables()
		gradients = tf.gradients(self.loss, tvars)
		clipped_gradients, self.global_gradient_norm = tf.clip_by_global_norm(gradients, self.max_gradient)

		tf.summary.scalar("global_grad_norm", self.global_gradient_norm)

		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

		try:
			self.train_gradients = zip(clipped_gradients, tvars)
		except ValueError as ex:
			print("Unable to re combine vars and grads, vars: {:,}  grads: {:,}".format(
				len(tvars), len(clipped_gradients)
			), ex)
			exit(1)

		self.train_op = self.optimizer.apply_gradients(self.train_gradients)

		try:
			for g, cg in zip(gradients, clipped_gradients):
				name = "grads/" + g.name
				tf.summary.histogram("raw/" + name, g)
				tf.summary.histogram("clipped/" + name, cg)
		except ValueError as ex:
			print("Error hanging gradient histograms: ", ex)
			exit(1)

		# tf.summary.histogram("raw_gradients", gradients)
		# tf.summary.histogram("clipped_gradients", gradients)

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
			print("\t", var.name)

		# values for tensorboard
		# some nice logging

		tf.summary.scalar("loss", self.loss)
		tf.summary.scalar("confusion/false_negatives", self.false_negatives)
		tf.summary.scalar("confusion/true_positives", self.true_positives)
		tf.summary.scalar("confusion/false_positives", self.false_positives)
		tf.summary.scalar("confusion/true_negatives", self.true_negatives)

		self.add_summaries(self.probs, "probability")

		# add this to make sure the annoying thing is initialized like it should be...
		confusion = self.confusion

	@staticmethod
	def add_summaries(item, name):
		tf.summary.scalar(name + "_max", tf.reduce_max(item))
		tf.summary.scalar(name + "_min", tf.reduce_min(item))

	@ifnotdefined
	def onehot_labels(self):
		tf.one_hot(indices=tf.to_int32(self.targets),
				   depth=self.args.num_classes, dtype=self.targets.dtype)

	@define_scope
	def loss(self):
		# loss_weights = tf.ones([self.args.batch_size, self.args.seq_length])
		# self.loss = s2s.sequence_loss(split_logits, tf.to_int32(self.targets),
		#							loss_weights, name="compute_loss")
		print("\nSetting up loss:")
		# confusion = self.confusion

		self.logits = tf.squeeze(self.logits)

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

		return self._loss  # + self.false_negative_loss_scale_factor  # tf.losses.get_total_loss()  # self._loss

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

		print("Initializing loss weights:")

		print("\tLabel Ratio: ", self.label_ratio)

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

		self.false_negatives = tf.to_int32(tf.reduce_sum(false_negatives))
		self.true_positives = tf.to_int32(tf.reduce_sum(true_positives))

		# make sure no zeros end up in the weights matrix
		scale_factor = tf.cond(tf.greater(self.false_negative_loss_scale_factor, 0),
							   true_fn=lambda: self.false_negative_loss_scale_factor,
							   false_fn=lambda: 1.0)

		print("\tCapping minimum false negative scale factor at: ", self.args.label_ratio)
		self._loss_scale = tf.cond(tf.less(scale_factor, self.args.label_ratio),
						true_fn=lambda: tf.to_float(self.args.label_ratio),
						false_fn=lambda: scale_factor)

		tf.summary.scalar("loss_scale_used", self._loss_scale)

		# scale = tf.add(scale, self.label_ratio)

		weighted_fp = tf.multiply(false_positives, 1.0)
		weighted_tp = tf.multiply(true_positives, .2)

		weighted_tn = tf.multiply(true_negatives, 1.0)
		weighted_fn = tf.multiply(false_negatives, self._loss_scale)

		# the above weights will punish false negatives and reward true positives
		return [weighted_tp, weighted_tn, weighted_fn, weighted_fp]  # weights

	# @define_scope
	# def loss_scale_factor(self):
	# 	return tf.log(tf.to_float(tf.abs(self._fn)))

	# @define_scope(scope="print_scales")
	@property
	def loss_scale_factors(self):
		return {"fn": self.false_negative_loss_scale_factor, "actual: ": self._loss_scale}

	@staticmethod
	def loglog(item):
		return tf.log(tf.log(tf.to_float(item)))

	@staticmethod
	def fn_punish(number):
		# return number / 2.0
		return tf.log(number)

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

		value = tf.to_float(tf.abs(self.false_negatives))
		# value = tf.Variable(1.0, dtype=self.gpu_type, name="loss_scale_factor", trainable=False)
		# if its zero, return one (no punishment) since log(0) is undefined
		# this avoids NaN weights
		cond = tf.cond(tf.equal(value, 0.0),
					   true_fn=lambda: 1.0,
					   false_fn=lambda: self.fn_punish(value))
		tf.summary.scalar(name="loss_scale_computed", tensor=cond)
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

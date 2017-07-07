from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math

import sklearn as sk
import tensorflow as tf

from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq as s2s

import numpy as np

from decorators import define_scope

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

	def build_three_layers(self, use_highway=True):
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
		first = self.cell_fn(outer_size)  # , num_proj = middle_size)
		# middle = None

		# normalize the first layer before the highway
		# avg_prob = abs(self.args.input_keep_prob / self.args.output_keep_prob)
		# first = rnn.LayerNormBasicLSTMCell(outer_size, dropout_keep_prob = avg_prob)

		if use_highway:
			self.args.using_highway = True
			middle = highway(self.cell_fn(outer_size))
		else:
			middle = self.cell_fn(outer_size)
			self.args.using_highway = False

		last = self.cell_fn(outer_size)

		# dropout on first and last
		first = self.add_dropout(first, self.args.input_keep_prob, self.args.output_keep_prob)
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

	# this will only work for a binary	classification problem
	# def compute_confusion(self, logits, targets):
	# 	"""Get the confusion matric from the logits and targets"""
	#
	# 	# logits => [batch_size, seq_length, vocab_size/num classes]
	# 	# targets => [batch_size, seq_length]
	#
	# 	# unroll the logits to [batch_size * seq_length, vocab_size]
	# 	# flatten targets to [ batch_size * seq_length ]
	# 	logits = tf.reshape(tf.to_float(logits), [-1, self.args.vocab_size])
	# 	targets = tf.reshape(tf.to_int32(targets), [-1])
	#
	# 	is_label_one = tf.cast(targets, dtype=tf.bool)
	# 	is_label_zero = tf.logical_not(is_label_one)
	#
	# 	try:
	# 		correct_pred = tf.nn.in_top_k(logits, targets, 1, name="correct_answer")
	# 	except ValueError as ex:
	# 		print("Unable to get correct pred:", ex)
	# 		exit(1)
	# 	false_pred = tf.logical_not(correct_pred)
	#
	#
	#
	# 	# self.precision, self.precision_update = tf.contrib.metrics.streaming_precision(correct_pred, targets)
	# 	# self.accuracy, self.accuracy_update = tf.contrib.metrics.streaming_accuracy(correct_pred, targets)
	# 	# self.recall, self.recall_update = tf.contrib.metrics.streaming_recall(correct_pred, targets)
	# 	# # ret["precision_"] = pred_update
	#
	# 	ret = {"precision": self.precision, "accuracy": self.accuracy, "recall": self.recall}
	#
	# 	return ret

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

		print("\nSetting self.lr = {:.5}".format(args.learning_rate))

		# self.lr = tf.Variable(args.learning_rate, name="lr", dtype=tf.float32)
		self._lr = None

		print("Cell type is: ", args.model)
		print("Batch size is: ", args.batch_size)

		def local_cell_fn(size):
			return rnn.LSTMCell(size)

		self.cell_fn = local_cell_fn
		# self.cell_fn = rnn.GRUCell
		# self.cell_fn = rnn.IntersectionRNNCell
		# self.cell_fn = rnn.NASCell

		with tf.name_scope("Cells"):
			print("Building cells of size: ", args.rnn_size)
			cells = self.build_cells()
			print("\nCells:")
			print("\tSquishing {} cells into one".format(len(cells)))
			for c in cells:
				print("\t", c)

			self.cell = rnn.MultiRNNCell(cells, state_is_tuple=True)
			cell = self.cell

		print("Setting self.initial_state based on batch size: ", self.args.batch_size)
		self.initial_state = cell.zero_state(self.args.batch_size, self.gpu_type)

		print("Setting self.num_batches")
		self.num_batches = num_batches

		# all teh data for the epoch
		# all of these are pinned to the gpu
		# self.all_input_data = None
		# self.all_target_data = None
		self._global_step = None
		self.inc_step = tf.assign_add(self.global_step, 1, use_locking=True, name="inc_global_step")
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

		# dropout beta testing: double check which one should affect next line
		if training and args.output_keep_prob:
			inputs = tf.nn.dropout(inputs, args.output_keep_prob)
		# processing inputs on cpu
		# with tf.device("/cpu:0"):

		output, last_state = self.build_outputs(inputs)

		print("Getting logits")

		# the final layers
		# maps the outputs	to [ vocab_size ] probs
		# self.logits = tf.contrib.layers.fully_connected(output, args.vocab_size)
		self.logits = tf.layers.dense(inputs=output, units=self.args.vocab_size)
		print("Logits shape: ", self.logits.shape)

		# both of these are for sampling
		with tf.name_scope("probabilities"):
			print("Getting probs")
			self.probs = tf.nn.softmax(self.logits, name="probs_softmax")

		with tf.name_scope("hardmax"):
			print("Getting hardmax")
			self.hardmax = tf.squeeze(s2s.hardmax(self.logits))
			self.single_hardmax = tf.argmax(self.hardmax)

		# we assume predicting one at a time for now
		# TODO rewite to be able to predict N number of seqs at once
		if not self.is_training:
			with tf.name_scope("predict_index"):
				# set predict to the right series of operations
				self.predict = tf.squeeze(self.probs)

		# make into [ batch_size, seq_len, vocab_size ]
		# it should already be this size, but this forces tf to recognize
		# the shape
		logits_shape = [self.args.batch_size, self.args.seq_length, self.args.vocab_size]
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

		self.final_state = last_state

		with tf.name_scope("optimizer"):
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
			self.train_op = self.optimizer.minimize(self.loss)


		print("\nSetup learning rate decay:")
		print("\tlr: {}\n\tdecay every {} steps\n\tdecay rate: {}\n\tstaircase: {}"
			  .format(self.lr, self.num_batches // 2, self.args.decay_rate, True))



		# self.train_op = tf.contrib.layers.optimize_loss(
		# 	loss = self.loss,
		# 	global_step=tf.contrib.framework.get_global_step(),
		# 	clip_gradients=self.args.grad_clip,
		# 	learning_rate_decay_fn=None,
		# 	name="optimize",
		# 	colocate_gradients_with_ops=True,
		# 	learning_rate=self.lr_decay,
		# 	optimizer="Adagrad")

		print("\ntrainable_variables:")
		for var in tf.trainable_variables():
			print("\t", var)

		# values for tensorboard
		# some nice logging
		tf.summary.scalar("max_loss", tf.reduce_max(self.loss))
		tf.summary.scalar("min_loss", tf.reduce_min(self.loss))

		tf.summary.scalar("max_prob", tf.reduce_max(self.probs))
		tf.summary.scalar("min_prob", tf.reduce_min(self.probs))

		tf.summary.scalar("learning_rate", self.lr)



	# tf.summary.histogram('logits', self.logits)
	# tf.summary.histogram('loss', self.loss)
	# t
	# f.summary.scalar('train_loss', self.cost)



	@property
	def loss(self):
		with tf.name_scope("compute_loss"):

			# loss_weights = tf.ones([self.args.batch_size, self.args.seq_length])
			# self.loss = s2s.sequence_loss(split_logits, tf.to_int32(self.targets),
			#							loss_weights, name="compute_loss")
			if self._loss is None:
				print("\nSetting up loss")
				print("\tCalling confusion to hang its internals")
				confusion = self.confusion
				self._label_ratio = tf.Variable(self.args.label_ratio, name="label_ratio", trainable=False, dtype=self.gpu_type)
				print("\tLabel Ratio: ", self._label_ratio)
				onehots = tf.one_hot(indices=tf.to_int32(self.targets),
								 depth=self.args.vocab_size, dtype=self.targets.dtype)

				# this will weight so that labels of "1" are more important
				# self._label_weight = tf.Variable(1.0/self._label_ratio, name="label_weight", dtype=tf.float32)
				# weight = tf.multiply(self._label_weight, tf.cast(tf.equal(self.targets, 1), tf.float32)) + 1

				weight = tf.ones([self.args.batch_size, self.args.seq_length])

				# self._loss = tf.losses.softmax_cross_entropy(onehot_labels=onehots, logits=self.logits)
				print("\tOnehots shape: ", onehots.shape)
				print("\tLogits shape: ", self.logits.shape)
				print("\tWeight shape: ", weight.shape)
				assert onehots.shape == self.logits.shape, "Logits shape != labels shape"

				preds = tf.nn.softmax(self.logits)
				preds = tf.argmax(preds, 2)
				preds = tf.to_float(preds)

				targets = tf.to_float(self.targets)

				assert preds.shape == targets.shape, "Preds shape != targets sghape"

				diff = tf.losses.absolute_difference(labels=targets, predictions=preds, reduction=tf.losses.Reduction.NONE)

				self._abs_diff = tf.count_nonzero(diff)

				# cast all to in
				preds = tf.to_int32(preds)
				targets = tf.to_int32(targets)
				diff = tf.to_int32(diff)

				# all three are of shape => [batch_size, seq_length]
				self._tp = tf.reduce_sum(tf.multiply(preds, targets))
				self._fp = tf.reduce_sum(tf.multiply(preds, diff))
				self._fn = tf.reduce_sum(tf.multiply(targets, diff))
				# now they are all scalars

				# avoid a division by zero
				tweak = tf.constant(1e-6, dtype=tf.float32)

				# cast all to bool , and then flip it
				bool_preds = tf.cast(preds, tf.bool)

				# where pred = 0 and targ = 1
				false_negs =  tf.cast(tf.multiply(targets, diff), tf.bool) #tf.logical_and(tf.logical_not(bool_preds), tf.cast(targets, tf.bool))

				# this will make all "NO" predictions where we wanted a "YES" be penalized by false_negative weight / 2
				# maybe good..?
				flipped_preds = tf.to_float(false_negs)

				flipped_preds = tf.multiply(flipped_preds, tf.multiply(tf.to_float(self._fn), 0.25))

				# everything thats not a false negative has a weight of one
				self._weights = tf.multiply(flipped_preds, tf.ones(flipped_preds.shape))

				# exit(1)
				self.other_precision = tf.divide(self._tp, self._tp + self._fp)
				self._loss = tf.losses.softmax_cross_entropy(onehot_labels=onehots, logits=self.logits)

			print("Returning loss")
			return self._loss


	@define_scope
	def cost(self):
		cost = self.loss
		return cost

	@property
	def lr_decay(self):
		if self._lr_decay is None:
			self._lr_decay = tf.train.exponential_decay(self.lr, global_step=self.global_step,
												   decay_steps=self.num_batches // 2, decay_rate=self.args.decay_rate,
											   staircase=True, name="decay_lr")
		return self._lr_decay

	@define_scope
	def lr(self):
		lr = tf.Variable(self.args.learning_rate, name="lr", dtype=tf.float32, trainable=False)
		return lr

	@define_scope
	def predictions(self):
		predictions = tf.reshape(tf.nn.softmax(self.logits), [-1, self.args.vocab_size])
		predictions = tf.argmax(predictions, 1)
		return predictions

	@define_scope
	def recall(self):
		targets = tf.reshape(self.targets, [-1])
		recall, _ = tf.metrics.recall(labels=targets, predictions=self.predictions)
		return recall

	@define_scope
	def accuracy(self):
		targets = tf.reshape(self.targets, [-1])
		accuracy, _ = tf.metrics.accuracy(labels=targets, predictions=self.predictions)
		return accuracy

	@define_scope
	def precision(self):
		targets = tf.reshape(self.targets, [-1])
		precision, _ = tf.metrics.precision(labels=targets, predictions=self.predictions)
		return precision

	@define_scope
	def confusion(self):
		confusion = {"accuracy": self.accuracy, "precision": self.precision, "recall": self.recall}
		return confusion

	@define_scope
	def global_step(self):
		global_step = tf.Variable(0, trainable=False, name="global_step")
		return global_step

	def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=0):
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

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq as s2s

import numpy as np

"""Build a RNN model """


class Model():
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
				print("Using the inference helper")
				start_tokens = tf.fill([self.args.seq_length], self.args.seq_length)
				end_token = 0
				print("\tState token: ", start_tokens.shape)
				print("\tEnd token: ", end_token)

				# decoder_helper = s2s.GreedyEmbeddingHelper(embedding,
				# start_tokens, end_token)
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
		with tf.device("/gpu:0"):
			args = self.args
			print("\nHanging GPU variables")
			all_shape = [self.num_batches, args.batch_size, args.seq_length]
			batch_shape = [args.batch_size, args.seq_length]
			
			print("\tall_shape: ", all_shape)
			print("\tbatch_shape: ", batch_shape)
			

			#all_s = tf.placeholder(self.gpu_type, [self.num_batches, args.batch_size, args.seq_length])

			print("\tsetting all input")
			self.all_input_data = tf.Variable(tf.zeros(all_shape, dtype=self.gpu_type),
											dtype=self.gpu_type, trainable=False,
											name="all_inputs")

			print("\tsetting all targets")
			self.all_target_data = tf.Variable(tf.zeros(all_shape, dtype=self.gpu_type),
											dtype=self.gpu_type, trainable=False,
											name="all_targets")
			
			
			self.step = tf.Variable(0, dtype=self.gpu_type, trainable=False, name="step")

			# data for each step
			self.input_data = tf.Variable(tf.zeros(batch_shape, dtype=self.gpu_type),
										dtype=self.gpu_type, name="batch_input",
										trainable=False)
			self.targets = tf.Variable(tf.zeros(batch_shape, dtype=self.gpu_type),
									dtype=self.gpu_type, name="batch_targets",
									trainable=False)
			
			self.step = tf.Variable(0, dtype=self.gpu_type, trainable=False, name="step")

			with tf.name_scope("inc_step"):
				self.inc_step = tf.assign_add(self.step, 1.0, name="inc_step")

			# grab the batch data for the current step
			with tf.name_scope("grab_step_data"):
				index = tf.to_int32(self.step, name="step_to_int")

				self.input_data = tf.assign(self.input_data, self.all_input_data[index])
				self.targets = tf.assign(self.targets, self.all_target_data[index])

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

	def build_cells(self):
		"""
		Build some RNN cells
		:return: a list of cells
		"""
		# only working number of layers right now
		if self.args.num_layers == 3:
			ret = self.build_three_layers()
		print("Done building cells")
		return ret

	def __init__(self, args, num_batches, training=True):
		""" init """
		# self.args = args
		
		if not training:
			print("Not training")
			print("Prev batch size: ", args.batch_size)
			print("Prex seq length: ", args.seq_length)
			#args.batch_size = 1
			#args.seq_length = 1

		self.args = args
		self.args.orig_batch_size = self.args.batch_size
		#self.args.batch_size = None
		#args.batch_size = None
		

		self.gpu_type = tf.float32
		self.is_training = training
		self.seq_length = self.args.seq_length

		self.lr = tf.Variable(0.0, trainable=False, name="lr")

		print("Cell type is: ", args.model)

		self.cell_fn = rnn.LSTMCell

		with tf.name_scope("Cells"):
			cells = self.build_cells()

			print("Squishing {} cells into one".format(len(cells)))
			print("Cells:  ", cells)
			self.cell = rnn.MultiRNNCell(cells, state_is_tuple=True)
			cell = self.cell

		print("Setting self.initial_state")
		self.initial_state = cell.zero_state(self.args.batch_size, self.gpu_type)
		
		print("Setting self.num_batches")
		self.num_batches = num_batches

		# all teh data for the epoch
		# all of these are pinned to the gpu
		self.all_input_data = None
		self.all_target_data = None
		self.step = None
		self.inc_step = None
		self.input_data = None
		self.targets = None
		

		print("Calling hang_gpu_variables")
		# assign values to the above variables
		self.hang_gpu_variables()

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
		self.logits = tf.contrib.layers.fully_connected(output, args.vocab_size)

		# both of these are for sampling
		with tf.name_scope("probabilities"):
			print("Getting probs")
			self.probs = tf.nn.softmax(self.logits, name="probs_softmax")

		with tf.name_scope("hardmax"):
			print("Getting hardmax")
			self.hardmax = tf.squeeze(s2s.hardmax(self.logits))

		# we assume predicting one at a time for now
		# TODO rewite to be able to predict N number of seqs at once
		if not self.is_training:
			with tf.name_scope("predict_index"):
				print("Probs shape: ", self.probs.shape)
				s = tf.squeeze(self.probs)
				s = tf.reshape(s, [args.batch_size, args.vocab_size])
				print("after: ", s.shape)

				self.predict = s  # tf.argmax(tf.squeeze(self.probs))
		# self.predict	= tf.reshape(s, [args.vocab_size])

		# make into [ batch_size, seq_len, vocab_size ]
		# it should already be this size, but this forces tf to recognize
		# the shape
		logits_shape = [args.batch_size, args.seq_length, args.vocab_size]
		split_logits = tf.reshape(self.logits, logits_shape)

		with tf.name_scope("compute_loss"):
			loss_weights = tf.ones([args.batch_size, args.seq_length])
			self.loss = s2s.sequence_loss(split_logits, tf.to_int32(self.targets),
										loss_weights, name="compute_loss")

		with tf.name_scope('cost'):
			self.cost = self.loss

		self.final_state = last_state

		with tf.name_scope("optimizer"):
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

		# grads, t_vars = zip(*self.optimizer.compute_gradients(self.cost))
		# grads, _ = tf.clip_by_global_norm(grads, self.args.grad_clip)
		# self.train_op = self.optimizer.apply_gradients(zip(grads, t_vars))

		with tf.name_scope("grad_clip"):
			gradients, variables = zip(*self.optimizer.compute_gradients(self.cost))
			# self.grads = gradients
			gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
		# self.clipped_grads = gradients
		with tf.name_scope("apply_grads"):
			self.train_op = self.optimizer.apply_gradients(zip(gradients, variables))

		with tf.name_scope("gradients"):
			if type(gradients) is list:
				for i in range(len(gradients)):
					print("Hanging grad histogram for: ", variables[i].name)
					tf.summary.histogram(variables[i].name, gradients[i])

		# instrument tensorboard
		# some nice logging
		tf.summary.scalar("max_loss", tf.reduce_max(self.loss))
		tf.summary.scalar("min_loss", tf.reduce_min(self.loss))

		tf.summary.scalar("max_prob", tf.reduce_max(self.probs))
		tf.summary.scalar("min_prob", tf.reduce_min(self.probs))

		tf.summary.scalar("learning_rate", self.lr)

		tf.summary.histogram('logits', self.logits)
		tf.summary.histogram('loss', self.loss)
		tf.summary.scalar('train_loss', self.cost)




	def sample(self, sess, chars, vocab, num=200, prime='The '):
		print("In sample")

		state = sess.run(self.cell.zero_state(1, tf.float32))
		print("Set state")
		for char in prime[:-1]:
			x = np.zeros((1, 1))
			x[0, 0] = vocab[char]
			feed = {self.input_data: x, self.initial_state: state}
			[state] = sess.run([self.final_state], feed)

		print("Built feed dict")

		ret = prime
		char = prime[-1]

		print("Kicking off the predictions...")
		for n in range(num):
			x = np.zeros((1, 1))
			x[0, 0] = vocab[char]
			feed = {self.input_data: x, self.initial_state: state}

			# logits = sess.run([self.logits], feed)

			# probs, hardmax = sess.run([self.probs, self.hardmax], feed)
			predict = sess.run([self.predict], feed)

			sample = predict

			print("sample: ", sample)
			prediction = chars[sample]
			ret += prediction
			char = prediction
		return ret

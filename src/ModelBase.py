from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import tensorflow as tf

from tensorflow.contrib import rnn
import numpy as np


from decorators import *

RANDOM_SEED = 10


class ModelException(Exception):
	pass


class ModelBase(object):

	def __init__(self, args, num_batches, is_training, gpu_type):
		assert args is not None
		assert type(num_batches) == int and num_batches > 0
		self._is_training = bool(is_training)

		# hang initial None data
		self.global_step = tf.Variable(-1, name="global_step", trainable=False)

		# the type of all variables through the system.
		# must be a float
		assert isinstance(gpu_type, tf.DType) and gpu_type.is_floating, "gou_type must be floating"
		self.gpu_type = gpu_type

		try:
			self.num_batches = int(num_batches)
		except Exception as ex:
			raise ModelException("Invalid num batches:  {}".format(ex))

		try:
			self.embedding_size = int(args.embedding_size)
		except Exception as ex:
			raise ModelException("Invalid embedding size:  {}".format(ex))

		try:
			self.seq_length = int(args.seq_length)
		except Exception as ex:
			raise ModelException("Invalid sequence length:  {}".format(ex))

		try:
			self.max_gradient = float(args.max_gradient)
		except Exception as ex:
			raise ModelException("Invalid max gradient value:  {}".format(ex))

		self.cell_fn = self.assign_cell_fn(args.model)

		try:
			self.max_gradient = float(args.max_gradient)
		except Exception as ex:
			raise ModelException("Invalid max gradient value:  {}".format(ex))

		try:
			self.vocab_size = float(args.vocab_size)
		except Exception as ex:
			raise ModelException("Invalid vocab size value:  {}".format(ex))

		try:
			self.num_classes = int(args.num_classes)
		except Exception as ex:
			raise ModelException("Invalid num classes value:  {}".format(ex))



		# do last, in case anything changes
		self.args = args



	@staticmethod
	def get_logits(input, output_size, name="logits"):
		return tf.layers.dense(inputs=input, units=output_size, name=name)

	@staticmethod
	def get_embedding(vocab_size, embedding_size, input_data, name="embedding"):
		shape = [vocab_size, embedding_size]
		embedding = tf.get_variable(name, shape, trainable=False)
		inputs = tf.nn.embedding_lookup(embedding, tf.to_int32(input_data))
		return embedding, inputs

	@staticmethod
	def assign_cell_fn(model):
		model = str(model).lower().strip()
		if model == "gru":
			print("Using GRU Cell")
			return rnn.GRUCell
		elif model == "nas":
			print("Using NAS Cell")
			return rnn.NASCell
		elif model == "glstm":
			print("Using glstm cell")
			return rnn.GLSTMCell
		elif model == "lstm":
			print("Using LSTM Cell")
			return rnn.LSTMCell
		else:
			raise ModelException("Invalid model type specified: {}".format(model))

	@property
	def is_training(self):
		return self._is_training

	@staticmethod
	def cluster(sequence):
		"""
		:param sequence: iterable
		:return: changed_sequence, number_changed
		"""
		sequence = np.array(sequence).flatten()
		labels_changed = 0
		for i in range(1, len(sequence) - 2):
			before = sequence[i-1]
			after = sequence[i+1]
			if (before == 1 == after) and sequence[i] == 0:
				sequence[i] = 1
				labels_changed += 1
			elif (before == 0 == after) and sequence[i] == 1:
				sequence[i] = 0
				labels_changed += 1
		return sequence, labels_changed

	@staticmethod
	def loglog(tensor):
		"""
		Calculate the log log base e of an item
		:param tensor:
		:return tensor
		"""
		return tf.log(tf.log(tf.to_float(tensor)), name="loglog")


	@staticmethod
	def _apply_rnn_dropout(cell, in_prob=1.0, out_prob=1.0,
						   state_prob=1.0, variational=False,
						   input_size=None, seed=RANDOM_SEED,
						   dtype=None):
		"""
		Given a cell and params, apply rnn dropout to the cell
		:param cell:
		:param in_prob:
		:param out_prob:
		:param state_prob:
		:param variational:
		:param input_size:
		:param seed:
		:param dtype:
		:return: Cell wrapped in rnn.DropoutWrapper
		"""

		assert type(state_prob) == float
		assert type(in_prob) == float
		assert type(out_prob) == float
		if variational:
			assert input_size is not None, "Must provide input_size with variational: {}".format(input_size)
			assert dtype is not None, "Must provide dtype with variational: {}".format(dtype)

		return rnn.DropoutWrapper(cell, input_keep_prob=in_prob,
								  output_keep_prob=out_prob, state_keep_prob=state_prob,
								  variational_recurrent=variational, input_size=input_size,
								  seed=seed, dtype=dtype)

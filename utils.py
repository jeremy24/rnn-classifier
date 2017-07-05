from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import codecs
import os
import collections
import math
import time
import itertools
import cProfile
import multiprocessing as multi
import concurrent.futures as cf
# from multiprocessing import Process, RawValue, Lock
# from multiprocessing.dummy import Pool as ThreadPool
from six.moves import cPickle
import numpy as np


class TextLoader(object):
	def __init__(self, data_dir, save_dir, batch_size, seq_length,
				 encoding='utf-8', todo=1000000, labeler_fn=None):
		self.data_dir = data_dir
		self.batch_size = batch_size
		self.seq_length = seq_length
		self.encoding = encoding
		self.labeler_fn = None

		self.vocab = dict()
		self.chars = dict()
		self.reverse_vocab = dict()
		self.vocab_size = 0
		self.tensor = None
		self.labels = list()
		self.num_chars = 0
		self.num_classes = 0
		self.num_batches = 0
		self.batches = None
		self.test_batches = None

		np.random.seed(int(time.time()))

		if labeler_fn is None:
			self.labeler_fn = lambda l: [str(x).isalpha() for s in l]
		else:
			print("User provided a labeler fn:", labeler_fn)
			self.labeler_fn = labeler_fn

		input_file = os.path.join(data_dir, "input.txt")
		vocab_file = os.path.join(save_dir, "vocab.pkl")
		tensor_file = os.path.join(save_dir, "data.npy")
		train_file = os.path.join(save_dir, "train_batches.npy")
		test_file = os.path.join(save_dir, "test_batches.npy")

		if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
			print("Preprocessing data")
			self.preprocess(input_file, vocab_file,
							tensor_file, train_file, test_file, todo=todo)
		else:
			print("loading preprocessed files")
			self.load_preprocessed(vocab_file, tensor_file,
								   train_file, test_file)

		self.reset_batch_pointer()

	def preprocess_helper(self, seq, raw_str):
		"""Take in ALL the x data and return {x: list, y: list}"""
		# encoded_seq = np.zeros(len(seq), dtype=np.uint16)
		self.chars = set()
		# seq_set = set(seq)

		print(type(seq))

		start = time.time()

		seq = np.array(seq)
		ord_mapper = np.vectorize(ord)
		encoded_seq = ord_mapper(seq)
		self.chars = np.unique(seq)
		for x in self.chars:
			self.vocab[x] = ord(x)

		print("Extracted out all chars, have: ", len(self.vocab), " Took: ", time.time()-start)
		labels = np.array(self.labeler_fn(raw_str), dtype=np.uint16)
		encoded_seq = np.array(encoded_seq, dtype=np.uint16)
		labels = np.ndarray.flatten(np.array(labels))
		assert len(encoded_seq) == len(labels), "Lens don't match {} != {}".format(len(chars), len(labels))
		assert len(self.chars) == len(self.vocab), "char and vocab lens mismatch"
		return {"x": encoded_seq, "y": labels}

	def preprocess(self, input_file, vocab_file,
				   tensor_file, train_file, test_file, todo=float("inf")):

		with codecs.open(input_file, "r", encoding=self.encoding) as f:
			data = f.read()

		self.chars = list()
		self.vocab = dict()
		self.reverse_vocab = dict()
		self.vocab_size = 0

		min_percent = 0.05

		if todo < len(data) * min_percent:
			print("todo of {:,} is less than {}% of {:,}, changing..."
				  .format(todo, int(min_percent * 100), len(data)))

			todo = len(data) * min_percent
			todo = int(todo)

		print("Preprocessing {:,} items from data".format(todo))
		print("Trimming data to length of todo")

		data = data[:todo]
		# flatten = lambda l: [item for sublist in l for item in sublist]
		start = time.time()
		seqs = list(data)

		print("Data length: {:,}".format(len(data)))
		print("Sequences converted to lists after: ", time.time() - start)

		# make sure its flat
		seqs = np.ndarray.flatten(np.array(seqs))
		print("Flattened seqs")

		pool_s = time.time()

		print("Starting preprocess {:,} items"
			  .format(len(seqs)))

		preprocess_data = self.preprocess_helper(seqs, data)
		encoded = preprocess_data["x"]
		labels = preprocess_data["y"]

		print("Labels generated in {:,.3f}".format(time.time() - pool_s))

		# make a reverse vocab lookup dict
		self.reverse_vocab = {v: k for k, v in self.vocab.items()}

		# drop any dupes
		self.chars = list(set(self.chars))

		assert len(self.chars) == len(self.vocab.items())
		assert len(encoded) == len(labels)

		# using uint 16 to save space and because we wont have more than 60k diff chars

		self.tensor = np.array(encoded, dtype=np.uint16)
		self.labels = np.array(labels, dtype=np.uint16)

		num_true_labels = np.sum(self.labels)

		print("Have {:,} true labeled chars out of {:,}  {:.4f}%"
			  .format(num_true_labels, len(self.labels),
					  num_true_labels / len(self.labels) * 100.0))

		assert len(self.labels) == len(self.tensor)

		# total number of chars in sample
		self.num_chars = len(self.tensor)

		# this is an "alias" to the number of classes in the problem
		self.vocab_size = len(set(self.labels))
		self.num_classes = self.vocab_size
		print("Processing took {:.3f}  vocab size: {}"
			  .format(time.time() - start, self.vocab_size))

		print("Dumping vocab to file...")
		with open(vocab_file, 'wb') as f:
			cPickle.dump(self.chars, f)

		print("Saving tensor file...")
		np.save(tensor_file, self.tensor)
		self.create_batches(train_file, test_file)

	def load_preprocessed(self, vocab_file, tensor_file, train_file, test_file):
		with open(vocab_file, 'rb') as f:
			self.chars = cPickle.load(f)
		self.vocab_size = len(self.chars)
		self.vocab = dict(zip(self.chars, range(len(self.chars))))
		print("Loading in preprocessed tensor file...")
		self.tensor = np.load(tensor_file)
		print("Tensor loaded")
		self.num_batches = int(self.tensor.size / (self.batch_size *
												   self.seq_length))
		self.batches = np.load(train_file)
		self.test_batches = np.load(test_file)

	def to_gb(self, num):
		return round(num / math.pow(2, 30), 3)

	def sample_batches(self, size=15):

		batch = self.batches[0]

		#  batch[batch_num][x|y][index_in_seq]
		x = batch[0][0][:size]
		y = batch[1][0][:size]

		print("len: ", len(batch), len(batch[0]), len(batch[0][0]))
		print(len(x), len(y))
		print(x[0])

		z = [self.reverse_vocab[idx] for idx in x]
		item = np.matrix([z, y])

		print(item)

	# print("X: ", z)
	# print("Y: ", y)

	def create_batches(self, train_file, test_file):
		# this was changed to be set in preprocess
		self.num_batches = int(self.tensor.size / (self.batch_size *
												   self.seq_length))

		print("Batch size changed to {:,}  have {:,} batches"
			  .format(self.batch_size, self.num_batches))

		# When the data (tensor)is too small,
		# let's give them a better error message
		if self.num_batches == 0:
			assert False, "Not enough data. Make seq_length and batch_size small."

		print("Total self.tensor size: ", self.to_gb(self.tensor.nbytes), "GB")

		num_examples = self.num_batches * self.batch_size * self.seq_length

		# xdata = self.tensor[:num_examples]
		# ydata = self.labels[:num_examples]

		xdata = self.tensor[:num_examples]
		ydata = self.labels[:num_examples]

		assert len(ydata) == len(xdata), "Data lengths don't match: {} != {}".format(len(xdata), len(ydata))

		# "alias" vocab size to really mean the number of labels
		self.vocab_size = len(list(set(ydata)))
		self.num_labels = self.vocab_size

		print("\nHave {} different labels".format(self.vocab_size))

		# dont misalign the labels for now
		# xdata = xdata[1:-2]
		# ydata[:-1] = ydata[1:]
		# ydata[-1] = ydata[0]

		assert len(ydata) == len(xdata)

		x_batches = np.split(xdata.reshape(self.batch_size, -1),
							 self.num_batches, 1)
		y_batches = np.split(ydata.reshape(self.batch_size, -1),
							 self.num_batches, 1)

		self.batches = list()
		sums = list()
		dropped = 0

		for i in range(len(y_batches)):
			x, y = x_batches[i], y_batches[i]

			y = np.array(y)
			if np.sum(np.ndarray.flatten(y)) < 50:
				dropped += 1
				continue
			sums.append(np.sum(y))
			item = np.array([x, y])
			item.flags.writeable = False
			self.batches.append(item)
		print("Dropped", dropped, "batches")
		print("avg:", np.mean(sums), "  min:", np.min(sums), "  max: ", np.max(sums))
		batch_members = len(y_batches[0][0]) * len(y_batches[0])
		perc = [np.mean(sum) / batch_members for sum in sums]
		print("avg:", np.mean(perc), "  min:", np.min(perc),
			  "  max: ", np.max(perc), "  median: ", np.median(perc))
		# exit(1)

		self.batches = np.array(self.batches)
		print("Batches built. Shuffling...")
		np.random.shuffle(self.batches)
		size = len(self.batches)
		test_size = int(size * 0.20)

		# set some data
		self.test_batches = self.batches[:test_size]
		self.batches = self.batches[test_size:]
		self.num_batches = len(self.batches)

		# save the batches to disk
		np.save(train_file, self.batches)
		np.save(test_file, self.test_batches)

		print("Build batches done...")
		self.sample_batches()

		print("Batches: {}	Test Batches:  {}"
			  .format(len(self.batches), len(self.test_batches)))

	def next_batch(self):
		# x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
		item = self.batches[self.pointer]
		# make immutable
		self.pointer += 1
		if self.pointer == self.num_batches:
			self.reset_batch_pointer()
		return item[0], item[1]

	def reset_batch_pointer(self):
		print("Reseting batch pointer...")
		self.pointer = 0
		print("Shuffling the batches...")
		np.random.shuffle(self.batches)

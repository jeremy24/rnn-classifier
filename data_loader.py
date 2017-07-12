from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import codecs
import os
import math
import time
import re
# from multiprocessing import Process, RawValue, Lock
# from multiprocessing.dummy import Pool as ThreadPool
from six.moves import cPickle
import numpy as np

from decorators import *


class TextLoader(object):
	def __init__(self, data_dir, save_dir, batch_size, seq_length,
				 encoding='utf-8', todo=1000000,
				 labeler_fn=None, is_training=False):
		self.data_dir = data_dir
		self.batch_size = batch_size
		self.seq_length = seq_length
		self.encoding = encoding
		self.labeler_fn = None
		self.save_dir = save_dir
		self.pointer = 0

		self.vocab = dict()
		self.chars = dict()
		self.reverse_vocab = dict()
		self.vocab_size = 0
		self.tensor = None
		self.labels = list()
		self.num_chars = 0
		self.num_classes = 0

		self._test_batches = None
		self._train_batches = None

		self.num_batches = 0
		self.batches = list()
		self.ratio = None

		self.replace_multiple_spaces = True
		self.is_training = bool(is_training)

		# make it predictably random
		# np.random.seed(int(time.time()))
		np.random.seed(5)

		if labeler_fn is None and self.is_training:
			print("\nNO LABELER FUNCTION PROVIDED\n")
			self.labeler_fn = lambda l: [str(s).isalpha() for s in l]
		elif not self.is_training:
			print("\nLabeler function not needed since model is not training")
			# This will now throw an error if something tries to call it
			# since we should not ever be labeling data if not training
			# at least for now
			self.labeler_fn = None
		else:
			print("User provided a labeler fn:", labeler_fn)
			self.labeler_fn = labeler_fn

		input_dir = "./inputs"

		if self.is_training:
			if self.have_saved_data():
				self.load_preprocessed()
			else:
				self.preprocess(input_dir, todo=todo)
		else:
			if self.have_saved_data():
				self.load_preprocessed()
			else:
				print("\nWe are not training and we have no preprocessed data!\n")
				assert False

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
		assert len(encoded_seq) == len(labels), "Lens don't match {} != {}".format(len(encoded_seq), len(labels))
		assert len(self.chars) == len(self.vocab), "char and vocab lens mismatch"
		return {"x": encoded_seq, "y": labels}

	def preprocess(self, input_path, todo=float("inf")):

		files = os.listdir(input_path)
		files = [os.path.join(input_path, filename) for filename in files if filename[0] != "."]
		print("Files: ", files)

		data = None

		print("\nProcessing files:")
		for filename in files:
			print("\t{}".format(filename))
			data = data + "\n\n" if data is not None else ""
			with open(filename, "r") as f:
				for line in f:
					data += line
			# with codecs.open(filename, "r", encoding=self.encoding) as f:
			# 	data += f.read()

		print("\n")

		self.chars = list()
		self.vocab = dict()
		self.reverse_vocab = dict()
		self.vocab_size = 0

		min_percent = 1.00  # 0.20

		if todo < len(data) * min_percent:
			print("todo of {:,} is less than {}% of {:,}, changing..."
				  .format(todo, int(min_percent * 100), len(data)))

			todo = len(data) * min_percent
			todo = int(todo)

		print("Preprocessing {:,} items from data".format(todo))
		print("Replacing spaces: {}".format(self.replace_multiple_spaces))
		print("Trimming data to length of todo")

		if self.replace_multiple_spaces == True:
			print("\nStripping spaces")
			print("\tBefore: ", len(data))
			data = re.sub(r"[ ]{2,}", " ", data)
			print("\tAfter: ", len(data))

		data = data[:todo]
		# flatten = lambda l: [item for sublist in l for item in sublist]
		start = time.time()
		seqs = list(data)

		# make sure its flat
		seqs = np.ndarray.flatten(np.array(seqs))

		label_start = time.time()

		print("Starting preprocess {:,} items"
			  .format(len(seqs)))

		preprocess_data = self.preprocess_helper(seqs, data)
		encoded = preprocess_data["x"]
		labels = preprocess_data["y"]

		print("Labels generated in {:,.3f}".format(time.time() - label_start))

		# make a reverse vocab lookup dict
		self.reverse_vocab = {v: k for k, v in self.vocab.items()}

		# drop any dupes
		self.chars = list(set(self.chars))

		assert len(self.chars) == len(self.vocab)
		assert len(encoded) == len(labels)

		# using uint 16 to save space and because we wont have more than 60k diff chars

		# self.tensor = np.array(encoded, dtype=np.uint16)
		# self.labels = np.array(labels, dtype=np.uint16)

		self.tensor = encoded
		self.labels = labels


		num_true_labels = np.sum(self.labels)

		self.ratio = num_true_labels / len(self.labels)

		print("Have {:,} true labeled chars out of {:,}  {:.4f}%"
			  .format(num_true_labels, len(self.labels), self.ratio * 100.0))

		self.ratio = 1.0 / self.ratio

		assert len(self.labels) == len(self.tensor)

		# total number of chars in sample
		self.num_chars = len(self.tensor)

		# this is an "alias" to the number of classes in the problem
		self.vocab_size = len(self.vocab)
		self.num_classes = len(set(self.labels))
		print("Processing took {:.3f}  vocab size: {}"
			  .format(time.time() - start, self.vocab_size))

		self.save_vocab_data()
		self.create_batches()


	@staticmethod
	@format_float(precision=3)
	def to_gb(num):
		return num / math.pow(2, 30)

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

	def trim_data(self):
		# chop off the end to make sure we have an even number of items
		size = self.seq_length * self.batch_size
		chop_line = (len(self.tensor) % size)
		chop_line = len(self.tensor) - chop_line
		print("size: {:,}  len: {:,} chopline: {:,}".format(size, len(self.tensor), chop_line))
		self.tensor = self.tensor[:chop_line]
		self.labels = self.labels[:chop_line]
		assert len(self.tensor) == len(self.labels)
		self.num_batches = len(self.tensor) // size

	def create_batches(self):
		self.trim_data()
		print("Batch size: {:,}  have {:,} batches"
			  .format(self.batch_size, self.num_batches))

		# When the data (tensor)is too small,
		# let's give them a better error message
		if self.num_batches < 1:
			assert False, "Not enough data. Make seq_length and batch_size small."

		print("Total self.tensor size: ", self.to_gb(self.tensor.nbytes), "GB")

		print("Num Batches to make: ", self.num_batches)

		xdata = np.array(self.tensor, dtype=np.uint16)
		ydata = np.array(self.labels, dtype=np.uint16)

		assert len(ydata) == len(xdata), "Data lengths don't match: {} != {}".format(len(xdata), len(ydata))

		self.vocab_size = len(self.vocab)
		self.num_classes = len(set(self.labels))

		print("\nHave {} different labels".format(self.vocab_size))

		# dont misalign the labels for now
		# xdata = xdata[1:-2]
		# ydata[:-1] = ydata[1:]
		# ydata[-1] = ydata[0]

		assert len(ydata) == len(xdata)

		print("len: ", len(xdata))

		# x_batches = np.split(xdata.reshape(self.batch_size, -1),
		# 					 self.num_batches, 1)
		# y_batches = np.split(ydata.reshape(self.batch_size, -1),
		# 					 self.num_batches, 1)


		size = int( len(xdata) / (self.seq_length * self.batch_size))

		print("Splitting {} into {} chunks".format(len(xdata), size))
		x_batches = np.split(xdata, size)
		x_batches = [ np.split(x, self.seq_length) for x in x_batches]

		z = x_batches[0]
		print([1 for x in z[0] if x == ord("\n")])

		print("".join([chr(a) for a in z[0]]))
		print("".join([chr(a) for a in z[1]]))
		print("".join([chr(a) for a in z[2]]))
		print("".join([chr(a) for a in z[3]]))
		print("".join([chr(a) for a in z[4]]))
		exit(1)


		self.batches = list()
		sums = list()
		dropped = 0
		# only drop up to a quarter of the batches
		max_drop = len(x_batches) / 4

		for i in range(len(y_batches)):
			x, y = x_batches[i], y_batches[i]

			y = np.array(y)
			# if less than 5% are highlighted, make eligible to dropping
			if dropped < max_drop and np.sum(np.ndarray.flatten(y)) < self.batch_size * .05:
				dropped += 1
				continue
			sums.append(np.sum(y))
			item = np.array([x, y])
			item.flags.writeable = False
			self.batches.append(item)
		print("Dropped", dropped, "batches")
		print("avg:", np.mean(sums), "  min:", np.min(sums), "  max: ", np.max(sums))
		batch_members = len(y_batches[0][0]) * len(y_batches[0])
		perc = [np.mean(s) / batch_members for s in sums]
		print("avg:", np.mean(perc), "  min:", np.min(perc),
			  "  max: ", np.max(perc), "  median: ", np.median(perc))
		# exit(1)

		# self.batches = np.array(self.batches)
		print("Batches built. Shuffling...")
		# np.random.shuffle(self.batches)
		size = len(self.batches)
		test_size = int(size * 0.20)

		# set some data
		# self.test_batches = self.batches[:test_size]
		# self.batches = self.batches[test_size:]
		print("Build batches done...")
		self.sample_batches()

		print("Batches: {}	Test Batches:  {}"
			  .format(len(self.batches), len(self.test_batches)))
		self.save_test_train()
		if not self.have_saved_data():
			print("\nData failed to save!\n")
			exit(1)

	@property
	def test_batches(self):
		if self._test_batches is None:
			if not self.is_training:
				msg = "Don't have testing batches from file"
				raise Exception(msg)
			size = int(self.num_batches * 0.20)
			self._test_batches = self.batches[:size]
		return self._test_batches

	@property
	def train_batches(self):
		if self._train_batches is None:
			if not self.is_training:
				msg = "Don't have training batches from file"
				raise Exception(msg)
			size = int(self.num_batches * 0.20)
			self._train_batches = self.batches[size:]
		return self._train_batches

	def save_test_train(self):
		train_file = os.path.join(self.save_dir, "train_batches.npy")
		test_file = os.path.join(self.save_dir, "test_batches.npy")
		try:
			# save the batches to disk
			np.save(train_file, self.batches)
			np.save(test_file, self.test_batches)
		except Exception as ex:
			print("Unable to save test/train data: ", ex)
			exit(1)

	def save_vocab_data(self):
		vocab_file = os.path.join(self.save_dir, "vocab.pkl")
		try:
			print("Dumping vocab to file...")
			with open(vocab_file, 'wb') as fout:
				cPickle.dump(self.chars, fout)
		except Exception as ex:
			print("Error saving vocab: ", ex)
			exit(1)

	def have_saved_data(self):
		train_file = os.path.join(self.save_dir, "train_batches.npy")
		test_file = os.path.join(self.save_dir, "test_batches.npy")
		vocab_file = os.path.join(self.save_dir, "vocab.pkl")
		return os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(vocab_file)

	def load_preprocessed(self):
		train_file = os.path.join(self.save_dir, "train_batches.npy")
		test_file = os.path.join(self.save_dir, "test_batches.npy")
		vocab_file = os.path.join(self.save_dir, "vocab.pkl")

		with open(vocab_file, 'rb') as f:
			self.chars = cPickle.load(f)

		self.vocab_size = len(self.chars)
		self.vocab = dict(zip(self.chars, range(len(self.chars))))


		print("Loading in preprocessed test and train files...")
		print("Tensor loaded")
		self._train_batches = np.load(train_file)
		self._test_batches = np.load(test_file)
		assert len(self._test_batches) > 0
		assert len(self._train_batches) > 0
		self.batches = None
		self.num_batches = None

	def next_batch(self):
		# x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
		item = self.train_batches[self.pointer]
		# make immutable
		self.pointer += 1
		if self.pointer == len(self.train_batches):
			self.reset_batch_pointer()
		return item[0], item[1]

	def reset_batch_pointer(self):
		print("Reseting batch pointer...")
		self.pointer = 0
		print("Shuffling the batches...")
		# np.random.shuffle(self.batches)

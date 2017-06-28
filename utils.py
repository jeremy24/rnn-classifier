from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import codecs
import os
import collections
import math
import itertools
from six.moves import cPickle
import numpy as np


class TextLoader():
	def __init__(self, data_dir, save_dir, batch_size, seq_length, encoding='utf-8', todo=1000000):
		self.data_dir = data_dir
		self.batch_size = batch_size
		self.seq_length = seq_length
		self.encoding = encoding

		input_file = os.path.join(data_dir, "input.txt")
		vocab_file = os.path.join(save_dir, "vocab.pkl")
		tensor_file = os.path.join(save_dir, "data.npy")
		train_file = os.path.join(save_dir, "train_batches.npy")
		test_file = os.path.join(save_dir, "test_batches.npy")


		if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
			print("Preprocessing data")
			self.preprocess(input_file, vocab_file, tensor_file, train_file, test_file, todo=todo)
		else:
			print("loading preprocessed files")
			self.load_preprocessed(vocab_file, tensor_file, train_file, test_file)
		self.reset_batch_pointer()
	
	#def preprocess_step(self, char):
	#	if char not in self.vocab


	def preprocess(self, input_file, vocab_file, 
			tensor_file, train_file, test_file, todo=float("inf")):
		
		with codecs.open(input_file, "r", encoding=self.encoding) as f:
			data = f.read()
		
		self.chars = list()
		self.vocab = dict()
		self.vocab_size = 0
		
		min_percent = .05

		if todo < len(data) * min_percent:
			print("todo of {:,} is less than {}% of {:,}, changing..."
					.format(todo, int(min_percent*100), len(data)))
			
			todo = len(data) * min_percent
			todo = int(todo)

		self.tensor = np.zeros(todo, dtype=np.uint16)


		i = 0
		print("Processing {:,} items from data".format(todo))
		for x in data:
			if i >= todo:
				break
			if x not in self.vocab:
				# assign a new id to that char
				self.vocab[x] = len(self.vocab) + 1
				self.chars.append(x)
				
			self.tensor[i] = self.vocab[x]
			i += 1

		self.vocab_size = len(self.vocab)
		print("Processing done.  Vocab size:", self.vocab_size)
		
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
		return round( num / math.pow(2, 30), 3)

	def create_batches(self, train_file, test_file):
		# this was changed to be set in preprocess
		self.num_batches = int(self.tensor.size / (self.batch_size *
												   self.seq_length))
		while self.num_batches > 5000:
			self.batch_size += 5
			self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))

		print("Batch size changed to {:,}  have {:,} batches".format(self.batch_size, self.num_batches))

		# When the data (tensor)is too small,
		# let's give them a better error message
		if self.num_batches == 0:
			assert False, "Not enough data. Make seq_length and batch_size small."

		print("Total self.tensor size: ", self.to_gb(self.tensor.nbytes), "GB")


		self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
		xdata = self.tensor
		ydata = np.copy(self.tensor)
		ydata[:-1] = xdata[1:]
		ydata[-1] = xdata[0]
		
		x_batches = np.split(xdata.reshape(self.batch_size, -1),
								  self.num_batches, 1)
		y_batches = np.split(ydata.reshape(self.batch_size, -1),
								  self.num_batches, 1)
		self.batches = list()
		for i in range(len(y_batches)):
			x, y = x_batches[i], y_batches[i]
			item = np.array([x, y])
			item.flags.writeable = False
			self.batches.append(item)

		self.batches = np.array(self.batches)
		print("Batches built. Shuffling...")
		np.random.shuffle(self.batches)
		size = len(self.batches)
		test_size = int(size * 0.20)
		self.test_batches = self.batches[:test_size]
		self.batches = self.batches[test_size:]
		self.num_batches = len(self.batches)

		np.save(train_file, self.batches)
		np.save(test_file, self.test_batches)

		print("Batches: {}	Test Batches:  {}"
			.format(len(self.batches), len(self.test_batches)))

	def next_batch(self):
		#x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
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






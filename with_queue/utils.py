import codecs
import os
import collections
from six.moves import cPickle
import numpy as np
import tensorflow as tf


## aliases to shorten code



class TextLoader():
	def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8', num_epochs=0):
		self.data_dir = data_dir
		self.batch_size = batch_size
		self.seq_length = seq_length
		self.encoding = encoding
		self.num_epochs = num_epochs

		input_file = os.path.join(data_dir, "input.txt")
		vocab_file = os.path.join(data_dir, "vocab.pkl")
		tensor_file = os.path.join(data_dir, "data.npy")
		
		self.input_file = input_file

		if  not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
			print("reading text file")
			self.preprocess(input_file, vocab_file, tensor_file)
		else:
			print("loading preprocessed files")
			self.load_preprocessed(vocab_file, tensor_file)
		self.create_batches()
		self.reset_batch_pointer()

	def preprocess(self, input_file, vocab_file, tensor_file):
		with codecs.open(input_file, "r", encoding=self.encoding) as f:
			 data = f.read()
		
		counter = collections.Counter(data)
		count_pairs = sorted(counter.items(), key=lambda x: -x[1])
		self.chars, _ = zip(*count_pairs)
		self.vocab_size = len(self.chars)
		self.vocab = dict(zip(self.chars, range(len(self.chars))))
		with open(vocab_file, 'wb') as f:
			cPickle.dump(self.chars, f)
		self.tensor = np.array(list(map(self.vocab.get, data)))
		np.save(tensor_file, self.tensor)

	def load_preprocessed(self, vocab_file, tensor_file):
		with open(vocab_file, 'rb') as f:
			self.chars = cPickle.load(f)
		self.vocab_size = len(self.chars)
		self.vocab = dict(zip(self.chars, range(len(self.chars))))
		self.tensor = np.load(tensor_file)
		self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))
		print("Preprocessed files loaded.")

	def create_batches(self):
		self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))

		batches_needed = self.num_batches * self.num_epochs

		# When the data (tensor) is too small,
		# let's give them a better error message
		if self.num_batches == 0:
			assert False, "Not enough data. Make seq_length and batch_size small."

		self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
		xdata = self.tensor
		ydata = np.copy(self.tensor)
		ydata[:-1] = xdata[1:]
		ydata[-1] = xdata[0]
		self.x_batches = np.split(xdata.reshape(self.batch_size, -1),
								  self.num_batches, 1)
		self.y_batches = np.split(ydata.reshape(self.batch_size, -1),
								  self.num_batches, 1)

		#self.batches =  list()
		#for i in range(len(self.x_batches)):
		#	self.batches.append((self.x_batches[i], self.y_batches[i]))
			#self.batches[i][1] = self.y_batches[i]

	
	def next_batch(self):

		x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
		#x, y = self.batches[self.pointer]
		self.pointer += 1
		left = len(self.x_batches) - self.pointer
		return x, y, left			

	def queue_next_batch(self):
		x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
		self.pointer += 1
		left = len(self.x_batches) - self.pointer
		if left == 0:
			self.reset_batch_pointer()
		item = {"x": x, "y":y}
		#print("Returning a queued batch:", item)
		return item
		
	def reset_batch_pointer(self):
		self.pointer = 0


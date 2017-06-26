import codecs
import os
import collections
from six.moves import cPickle
import numpy as np


class TextLoader():
	def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
		self.data_dir = data_dir
		self.batch_size = batch_size
		self.seq_length = seq_length
		self.encoding = encoding

		input_file = os.path.join(data_dir, "input.txt")
		vocab_file = os.path.join(data_dir, "vocab.pkl")
		tensor_file = os.path.join(data_dir, "data.npy")

		if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
			print("reading text file")
			self.preprocess(input_file, vocab_file, tensor_file)
		else:
			print("loading preprocessed files")
			self.load_preprocessed(vocab_file, tensor_file)
		self.create_batches()
		self.reset_batch_pointer()

	def preprocess(self, input_file, vocab_file, tensor_file, todo=5000000):
		with codecs.open(input_file, "r", encoding=self.encoding) as f:
			data = f.read()
		
		self.chars = list()
		self.vocab = dict()
		self.vocab_size = 0
		
		todo = len(data) if len(data) < todo else todo

		self.tensor = np.zeros(todo, dtype=np.uint16)
		
		idx = 0
		i = 0
		print("Processing {} items from data".format(todo))
		for x in data:
			if i >= todo:
				break
			if x not in self.vocab:
				self.vocab[x] = idx
				self.chars.append(x)
				idx += 1
			self.tensor[i] = self.vocab[x]
			i += 1
		
		self.vocab_size = len(self.vocab)
		print("Processing done.  Vocab size:", self.vocab_size)
		
		print("Dumping vocab to file...")
		with open(vocab_file, 'wb') as f:
			cPickle.dump(self.chars, f)
		
		print("Saving tensor file...")
		np.save(tensor_file, self.tensor)

	def load_preprocessed(self, vocab_file, tensor_file):
		with open(vocab_file, 'rb') as f:
			self.chars = cPickle.load(f)
		self.vocab_size = len(self.chars)
		self.vocab = dict(zip(self.chars, range(len(self.chars))))
		self.tensor = np.load(tensor_file)
		self.num_batches = int(self.tensor.size / (self.batch_size *
												   self.seq_length))

	def create_batches(self):
		self.num_batches = int(self.tensor.size / (self.batch_size *
												   self.seq_length))

		# When the data (tensor) is too small,
		# let's give them a better error message
		if self.num_batches == 0:
			assert False, "Not enough data. Make seq_length and batch_size small."

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






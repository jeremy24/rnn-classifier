from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import codecs
import os
import collections
import math
import time
import itertools
import multiprocessing as multi
# from multiprocessing import Process, RawValue, Lock
from multiprocessing.dummy import Pool as ThreadPool
from six.moves import cPickle
import numpy as np




class TextLoader(object ):
	def __init__(self, data_dir, save_dir, batch_size, seq_length, 
			encoding='utf-8', todo=1000000, labeler_fn=None, label_seq=False):
		self.data_dir = data_dir
		self.batch_size = batch_size
		self.seq_length = seq_length
		self.encoding = encoding
		self.labeler_fn = None
		self.label_whole_chunk = False

		np.random.seed(int(time.time()))

		if label_seq:
			self.label_whole_chunk = True	

		if labeler_fn is None:
			self.labeler_fn = lambda char: str(char).isalpha()
		else:
			print("User provided a labeler fn:", labeler_fn)
			self.labeler_fn = labeler_fn
			print(self.labeler_fn("a"), self.labeler_fn("1"))

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
	
	def preprocess_helper(self, seq):
		chars = list()
		labels = list()
		for char in seq:
			if char not in self.vocab:
				self.vocab[char] = ord(char)
				self.chars.append(char)
			chars.append(self.vocab[char])
			if self.label_whole_chunk == False:
				labels.append(self.labeler_fn(char))
		if self.label_whole_chunk:
			labels = self.labeler_fn(seq)
		assert len(chars) == len(labels), "Lens don't match {} != {}".format(len(chars), len(labels))
		labels = np.ndarray.flatten(np.array(labels))
		return { "x": chars, "y": labels }

	def preprocess(self, input_file, vocab_file, 
			tensor_file, train_file, test_file, todo=float("inf")):
		
		with codecs.open(input_file, "r", encoding=self.encoding) as f:
			data = f.read()
		
		self.chars = list()
		self.vocab = dict()
		self.reverse_vocab = dict()
		self.vocab_size = 0
		
		min_percent = .01

		if todo < len(data) * min_percent:
			print("todo of {:,} is less than {}% of {:,}, changing..."
					.format(todo, int(min_percent*100), len(data)))
			
			todo = len(data) * min_percent
			todo = int(todo)
	

		print("Preprocessing {:,} items from data".format(todo))		
	
		# give each worker 10 mil to do, dont exceed num of cpus
		# add one to avoid divide by zero
		
		per_worker = 5000000 if not self.label_whole_chunk else 1000000
		calc_num = todo // per_worker + 1
	
		num_workers = calc_num if calc_num <= multi.cpu_count() else multi.cpu_count()

		flatten = lambda l: [item for sublist in l for item in sublist]
	
		# to batch out the data
		stride = todo // num_workers
		
		start = time.time()

		# kickoff the preprocessing
		pool = ThreadPool(num_workers)

		# split the data into num_workers chunks
		str_chunks = [ data[i*stride:i+1*stride] for i in range(num_workers) ]

		# convert each chunk into a list of chars
		seqs = pool.map(list, str_chunks)
		
		pool.close()
		pool.join()

		pool = ThreadPool(num_workers)
		
		# if we dont want to label the whole seq,
		# then split it up so that that worker threads get a char
		# instead of a list of chars
		if self.label_whole_chunk == False:
			seqs = np.ndarray.flatten(np.array(seqs))
				
		# preprocessing each chunk
		data = pool.map(self.preprocess_helper, seqs)

		# if the user labeled whole chunks then data will
		# be a list of lists so we need to flatten in in case
		data = np.array(data)
		data = np.ndarray.flatten(data)

		pool.close()
		pool.join()

		ret = list()
		labels = list()

		# go thru each workers output and concat
		# it with the master output
		for batch in data:
			xs = batch["x"]
			ys = batch["y"]
			if type(xs) is np.ndarray:
				xs = xs.tolist()
			if type(ys) is np.ndarray:
				ys = ys.tolist()
			ret += xs
			labels += ys


		# make a reverse vocab lookup dict
		self.reverse_vocab = { v: k for k, v in self.vocab.items() }

		#for key in self.vocab:
		#	val = self.vocab[key]
		#	self.reverse_vocab[val] = key
		
		# drop any dupes
		self.chars = list(set(self.chars))

		assert len(self.chars) == len(self.vocab)
		assert len(ret) == len(labels)

		# flatten the results
		# ret = flatten(ret) if type(ret[0]) is list else ret
		# labels = flatten(labels) if type(labels[0]) is list else labels

		# using uint 16 to save space and becasue we wont have more than 60k diff chars
		self.tensor = np.array(ret, dtype=np.uint16)

		# to label the data
		self.labels = np.array(labels, dtype=np.uint8)

		print("Some labels: ", self.labels[:20])

		assert len(self.labels) == len(self.tensor)

		# print("Took ", time.time() - start, "using {} workers".format(num_workers))
		
		print("Tensor length: ", len(self.tensor))
		
		self.num_chars = len(self.tensor)

		self.vocab_size = len(self.vocab)
		print("Processing took {:.3f} using {} workers, vocab size: {}"
				.format(time.time()-start, num_workers, self.vocab_size))

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
		#while self.num_batches > 5000:
		#	self.batch_size += 5
		#	self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))

		print("Batch size changed to {:,}  have {:,} batches"
				.format(self.batch_size, self.num_batches))

		# When the data (tensor)is too small,
		# let's give them a better error message
		if self.num_batches == 0:
			assert False, "Not enough data. Make seq_length and batch_size small."

		print("Total self.tensor size: ", self.to_gb(self.tensor.nbytes), "GB")

		num_examples = self.num_batches * self.batch_size * self.seq_length

		self.tensor = self.tensor[:num_examples]
		self.labels = self.labels[:num_examples]

		xdata = self.tensor
		ydata = self.labels

		assert len(ydata) == len(xdata), "Data lengths don't match: {} != {}".format(len(xdata), len(ydata))

		# "alias" vocab size to really mean the number of labels
		self.vocab_size = len(list(set(ydata)))
		self.num_labels = self.vocab_size

		print("\nHave {} different labels".format(self.vocab_size))
		# scoot the data around so that x[0] is labeled by y[1]
		# then realign the data so the lengths are right and chop the ends
		#ydata[:-1] = xdata[1:]
		#ydata[-1] = xdata[0]
		
		## need to slice the front off ys and the end of xs

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






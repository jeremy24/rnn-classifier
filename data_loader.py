from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


import os
import math
import time
import re
import json
import glob
# from multiprocessing import Process, RawValue, Lock
# from multiprocessing.dummy import Pool as ThreadPool
from six.moves import cPickle
import numpy as np

from decorators import *
from process_real_data import process_ann_files

class TextLoader(object):
	def __init__(self, data_dir, save_dir, batch_size, seq_length,
				 encoding='utf-8', todo=1000000,
				 labeler_fn=None, is_training=False,
				 read_only=False, max_word_length=None,
				 using_real_data=False):
		self.data_dir = data_dir
		self.batch_size = batch_size
		self.seq_length = seq_length
		self.encoding = encoding
		self.labeler_fn = None
		self.save_dir = save_dir
		self.pointer = 0
		self.max_word_length = max_word_length
		self.using_real_data = using_real_data

		# self.vocab = dict()
		self.chars = dict()
		self.tensor = None
		self.labels = list()
		self.num_chars = 0
		self.num_classes = 0
		self.read_only = read_only
		self.label_ratio = None

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
		start = time.time()

		print("\tSeq type: ", type(seq))
		seq = np.array(seq)
		ord_mapper = np.vectorize(ord)
		encoded_seq = ord_mapper(seq)
		self.chars = np.unique(seq)

		print("Extracted out all chars, have: ", len(self.vocab), " Took: ", time.time() - start)
		labels = np.array(self.labeler_fn(raw_str, filepath=filepath), dtype=np.uint16)
		encoded_seq = np.array(encoded_seq, dtype=np.uint16)
		labels = np.ndarray.flatten(np.array(labels))
		assert len(encoded_seq) == len(labels), "Lens don't match {} != {}".format(len(encoded_seq), len(labels))
		assert len(self.chars) == len(self.vocab), "char and vocab lens mismatch"
		return {"x": encoded_seq, "y": labels}

	@staticmethod
	def trim_to_max_word_length(data, max_length):
		assert type(max_length) == int, "Max word length must be an int"
		assert type(data) == str, "Data passed to trim_to_max_word_length must be a string, got: {}".format(type(data))
		print("\nHave a max word length of {:,}".format(max_length))
		print("\tStarting length: {:,}".format(len(str(data))))
		local_data = str(data).split()
		keep = ""
		for x in local_data:
			if len(str(x)) <= max_length:
				keep += str(x) + " "
		print("\tNew length: {:,}".format(len(keep)))
		return keep

	def real_data_helper(self, dir_path):
		print("\tdirpath: [{}]  type: {}".format(dir_path, type(dir_path)))
		ext = ".labeled"
		folder = "labeled_data"
		assert os.path.exists(dir_path)
		replacement = chr(1)
		process_ann_files(dir_path, replace_char=replacement, ext=ext, folder=folder)
		print("\tDone processing ann files")
		labeled_dir = os.path.join(dir_path, folder)


		seq = ""
		seq_ = ""
		input_dir = dir_path + "/**/*.txt"
		print("\tInput files: ", input_dir)
		files =  glob.glob(input_dir, recursive=True)
		assert len(files) > 0, "Input directory is empty"
		print("\tProcessing {:,} input files".format(len(files)))
		for filename in files:
			with open(filename, "r") as fin:
				for line in fin:
					seq += line
			labels = filename.split("/")
			labels[-1] = folder + "/" + labels[-1] + ext
			labels = "/".join(labels)

			with open(labels, "r") as fin:
				for line in fin:
					seq_ += line

		seq_ = list(seq_)
		seq = list(seq)
		for i in range(len(seq_)):
			seq_[i] = seq_[i] == replacement

		seq = np.array(seq)
		labels = np.array(seq_)

		ord_mapper = np.vectorize(ord)
		encoded_seq = ord_mapper(seq)
		self.chars = np.unique(seq)

		assert len(encoded_seq) == len(labels)
		assert len(set(labels)) == 2
		# exit(1)
		return {"x": encoded_seq, "y": labels}

	def preprocess(self, input_path, todo=float("inf")):


		data = None
		start = time.time()

		print("\nProcessing files from input path: {}".format(input_path))

		# get labels for the data
		if self.using_real_data:
			print("\tUsing real data")
			preprocess_data = self.real_data_helper(input_path)
		else:
			print("\tUsing other data")
			files = os.listdir(input_path)
			files = [os.path.join(input_path, filename) for filename in files if filename[0] != "."]

			for filename in files:
				print("\t{}".format(filename))
				data = data + "\n\n" if data is not None else ""
				with open(filename, "r") as f:
					for line in f:
						data += line

			print("\n")

			if self.max_word_length is not None:
				data = self.trim_to_max_word_length(data, self.max_word_length)

			self.chars = list()
			# self.vocab = dict()
			# self.vocab_size = 0

			min_percent = .05  # 0.20

			if "MODEL_DATA_MIN_PERCENT" in os.environ:
				try:
					passed_value = float(os.environ["MODEL_DATA_MIN_PERCENT"])
					if 0.0 < passed_value <= 1.0:
						min_percent = passed_value
						print("Min percent passed in from env and was changed to: ", min_percent)
					elif 0.0 < passed_value <= 100.0:
						min_percent = passed_value / 100.0
						print("Min percent passed in from env and was changed to: ", min_percent)
					else:
						print("\nInvalid value passed in for min percent, not using:  ", passed_value)
				except ValueError:
					print("\nMin percent passed as env variable is not a valid float, not using it: ",
						  os.environ["MODEL_DATA_MIN_PERCENT"], "\n")

			if todo < len(data) * min_percent:
				print("todo of {:,} is less than {}% of {:,}, changing..."
					  .format(todo, int(min_percent * 100), len(data)))

				todo = len(data) * min_percent
				todo = int(todo)

			print("Preprocessing {:,} items from data".format(todo))
			print("Replacing spaces: {}".format(self.replace_multiple_spaces))
			print("Trimming data to length of todo")

			if self.replace_multiple_spaces:
				print("\nStripping multiple newlines")
				print("\tBefore: {:,}".format(len(data)))
				# data = re.sub(r"[\n]{3,}", "\n", data)
				data = re.sub(r"[\t]{2}", "\t", data)
				data = re.sub(r"[\t]{2}", "\t", data)
				data = re.sub(r"[\n]{2}", "\n", data)
				print("\tAfter:  {:,}".format(len(data)))

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
			print("Labels generated in {:,.3f}".format(time.time() - label_start))

		# if/else is done so grab the correct data out
		encoded = preprocess_data["x"]
		labels = preprocess_data["y"]



		# drop any dupes
		self.chars = list(set(self.chars))

		assert len(self.chars) == len(self.vocab)
		assert len(encoded) == len(labels)

		# these are what make_batches looks for
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
		self.num_classes = len(set(self.labels))

		print("Processing took {:.3f}  vocab size: {}"
			  .format(time.time() - start, self.vocab_size))

		self.save_vocab_data()
		self.create_batches()

	@property
	def vocab_size(self):
		"""
			A dynamic getter for the vocabulary size, just in case it changes
			:return: len(self.vocab)
		"""
		assert self.vocab is not None, "Cannot get vocab size, vocab is None"
		return len(self.vocab)

	@staticmethod
	@format_float(precision=3)
	def to_gb(num):
		return num / math.pow(2, 30)

	def sample_batches(self, size=30):
		print("\nSAMPLE")
		batch = self.test_batches[0]

		#  batch[batch_num][x|y][index_in_seq]
		x = batch[0][1][:size]
		y = np.array(batch[1][1][:size])

		assert len(batch) == 2, "Batch has more than x, y pairs in it"
		print("\tBatch size: {:,}".format(len(batch[0])))
		print("\tSeq length: {:,}".format(len(batch[0][0])))
		print("\tSample size: {:,}".format(size))

		assert len(x) == len(y)
		z = [self.reverse_vocab[idx] for idx in x]
		print("X: ", "".join(z).replace("\n", " "))
		print("Y: ", "".join([str(item) for item in y]))
		print("END SAMPLE\n")

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

		print("\nHave {} different labels".format(self.vocab_size))
		print("len: {:,}".format(len(xdata)))

		num_chunks = int(len(xdata) / (self.seq_length * self.batch_size))

		print("Splitting {} into {} chunks".format(len(xdata), num_chunks))
		print("{:,} / {:,} = {:,}".format(len(xdata), (self.seq_length * self.batch_size), num_chunks))
		print("seq len: {:,}  batch size: {:,}".format(self.seq_length, self.batch_size))

		x_batches = np.split(xdata, num_chunks)
		x_batches = [np.split(x, int(len(x) / self.seq_length)) for x in x_batches]

		y_batches = np.split(ydata, num_chunks)
		y_batches = [np.split(y, int(len(y) / self.seq_length)) for y in y_batches]

		print("\n{} batches of {} items with {} length strings\n".format(
			len(x_batches),
			len(x_batches[0]),
			len(x_batches[0][0])
		))

		# z = x_batches[0]
		# print("\nA sample of the data: (## signifies the boundaries between sequences)\n")
		# print("{}##{}##{}##{}##{}##".format("".join([chr(a) for a in z[0]]),
		# 		"".join([chr(a) for a in z[1]]),
		# 		"".join([chr(a) for a in z[2]]),
		# 		"".join([chr(a) for a in z[3]]),
		# 		"".join([chr(a) for a in z[4]])).replace("\t", "\t"))

		# only drop up to a quarter of the batches
		max_drop = len(x_batches) / 4

		# drop some sparsely labeled x,y sets to improve the ratio if we can
		self.batches, sums, dropped = self.drop_sparse(self.batch_size, max_drop, y_batches, x_batches)

		print("\nSums:")
		print("\tAvg: ", np.mean(sums), "\n\tMin: ", np.min(sums), "\n\tMax: ", np.max(sums))
		batch_members = len(y_batches[0][0]) * len(y_batches[0])
		percent = [round((np.mean(s) / batch_members) * 100, 3) for s in sums]
		print("\nLabel Ratios:")
		print("\n\tAvg:    ", round(np.mean(percent)), "%\n\tMin:    ", np.min(percent),
			  "%\n\tMax:    ", np.max(percent), "%\n\tMedian: ", np.median(percent), "%")

		# this save call will init both the train and test batch properties
		# on the object and will cause the data to be subdivided correctly
		# after this they return their local copy and will not touch the batches property
		# so after the call we set self.batches to None to prevent it being used
		self.save_test_train()
		self.batches = None

		self.sample_batches()

		print("Train Batches: {}	Test Batches:  {}"
			  .format(len(self.train_batches), len(self.test_batches)))

		print("Build batches done...")

		if not self.have_saved_data():
			print("\nData failed to save!\n")
			exit(1)

	@staticmethod
	def drop_sparse(batch_size, max_drop, y_batches, x_batches):
		print("\nDropping sparse labeled rows")
		data = list()
		sums = list()
		dropped = 0
		for i in range(len(y_batches)):
			x, y = x_batches[i], y_batches[i]
			y = np.array(y)
			# if less than 5% are highlighted, make eligible to dropping
			if dropped < max_drop and np.sum(np.ndarray.flatten(y)) < batch_size * .05:
				dropped += 1
				continue
			sums.append(np.sum(y))
			item = np.array([x, y])
			item.flags.writeable = False
			data.append(item)
		print("\tDropped", dropped, "batches")
		return data, sums, dropped

	@property
	def reverse_vocab(self):
		"""Make a reverse lookup dict for the vocabulary characters"""
		return {v: k for k, v in self.vocab.items()}

	@property
	def test_batches(self):
		"""The first time this is called it will split off its chunk of the batches"""
		if self._test_batches is None:
			if not self.is_training:
				msg = "Don't have testing batches from file"
				raise Exception(msg)
			size = int(self.num_batches * 0.20)
			self._test_batches = self.batches[:size]
		return self._test_batches if self._test_batches is not None else list()

	@property
	def train_batches(self):
		"""The first time this is called it will split off its chunk of the batches"""
		if self._train_batches is None:
			if not self.is_training:
				msg = "Don't have training batches from file"
				raise Exception(msg)
			size = int(self.num_batches * 0.20)
			self._train_batches = self.batches[size:]
		return self._train_batches if self._train_batches is not None else list()

	def save_test_train(self):
		"""Save train and test data to disk in .npy format"""
		train_file = os.path.join(self.save_dir, "train_batches.npy")
		test_file = os.path.join(self.save_dir, "test_batches.npy")
		try:
			np.save(train_file, self.train_batches)
			np.save(test_file, self.test_batches)
		except Exception as ex:
			print("Unable to save test/train data: ", ex)
			exit(1)

	def save_vocab_data(self):
		"""Save the vocabulary data to disk"""
		vocab_file = os.path.join(self.save_dir, "vocab.pkl")
		if self.read_only:
			print("\nNot saving vocab, in read only mode\n")
			return
		try:
			print("Dumping vocab to file...")
			with open(vocab_file, 'wb') as fout:
				cPickle.dump(self.chars, fout)
		except Exception as ex:
			print("Error saving vocab: ", ex)
			exit(1)

	def have_saved_data(self):
		"""Check if we have the appropriate saved data files"""
		train_file = os.path.join(self.save_dir, "train_batches.npy")
		test_file = os.path.join(self.save_dir, "test_batches.npy")
		vocab_file = os.path.join(self.save_dir, "vocab.pkl")
		if not os.path.exists(train_file):
			print("Don't have train file:", train_file)
			return False
		if not os.path.exists(test_file):
			print("Don't have test file:", test_file)
			return False
		if not os.path.exists(vocab_file):
			print("Don't have vocab file:", vocab_file)
			return False
		return True

	def load_preprocessed_vocab(self):
		"""
		 	Load in a saved vocab file
		"""
		try:
			vocab_file = os.path.join(self.save_dir, "vocab.pkl")
			with open(vocab_file, 'rb') as f:
				self.chars = cPickle.load(f)
				assert len(self.chars) > 0, "Loaded vocabulary is empty!"
		except Exception as ex:
			print("Unable to load preprocessed vocab file: ", ex)
			exit(1)

	@property
	def vocab(self):
		"""
			The vocab map, this is computed every call.
			It is NOT cached just in case the vocab changes
		"""
		ret = {}
		for x in self.chars:
			ret[x] = ord(x)
		return ret

	def load_preprocessed(self):
		"""
			Load in preprocessed text data, thiw will circumvent the normal
			way we initialize the train and test batch split by directly setting the values
		"""
		assert self.have_saved_data(), "Can't load preprocessed data, files are missing"

		print("Load preprocessed data:")
		train_file = os.path.join(self.save_dir, "train_batches.npy")
		test_file = os.path.join(self.save_dir, "test_batches.npy")

		self.load_preprocessed_vocab()

		# self.vocab_size = len(self.chars)


		print("\tloading test and train files...")
		self._train_batches = np.load(train_file)
		self._test_batches = np.load(test_file)
		print("\tTest size:   ", len(self._test_batches))
		print("\tTrain size:  ", len(self._train_batches))
		assert len(self._test_batches) > 0
		assert len(self._train_batches) > 0
		self.batches = None
		self.num_batches = None

		with open(os.path.join(self.save_dir, "hyper_params.json"), "r") as saved_args:
			saved = json.load(saved_args)
			try:
				self.num_batches = saved["num_batches"] or None
				self.label_ratio = saved["label_ratio"] or None
				self.num_classes = saved["num_classes"] or None
				self.num_chars = saved["num_chars"] or None
				print("\tRatio:       ", self.label_ratio)
				print("\tNum classes: ", self.num_classes)
			except KeyError as ex:
				print("data_loader is missing a saved params key: ", ex)
				exit(1)

	def next_batch(self):
		"""DEPRECATED"""
		# x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
		item = self.train_batches[self.pointer]
		# make immutable
		self.pointer += 1
		if self.pointer == len(self.train_batches):
			self.reset_batch_pointer()
		return item[0], item[1]

	def next_train_batch(self):
		"""Fetch teh next training batch"""
		item = self.train_batches[self.pointer]
		self.pointer += 1
		if self.pointer >= len(self.train_batches):
			self.reset_batch_pointer()
		return item[0], item[1]

	def next_test_batch(self):
		"""Fetch the next test batch"""
		item = self.test_batches[self.pointer]
		self.pointer += 1
		if self.pointer >= len(self.test_batches):
			self.reset_batch_pointer()
		return item[0], item[1]

	def reset_batch_pointer(self, quiet=False):
		"""Reset the batch pointer"""
		if not quiet:
			print("Reseting batch pointer...")
		self.pointer = 0

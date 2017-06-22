from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
from tensorflow.contrib import seq2seq as s2s

import numpy as np

"""Build a RNN model """

class Model():
	""" The RNN model """
	
	def add_dropout(self, cells, in_prob=1.0, out_prob=1.0):
		""" add dropout wrappers to cell[s], recursive """
		if type(cells) is not list:
			return rnn.DropoutWrapper(cells, input_keep_prob=in_prob,
					output_keep_prob = out_prob)
		else:
			ret = [ add_dropout_wrapper(cell, in_prob, out_prob) for cell in cells ]
			return ret

				

	
	def __init__(self, args, num_batches, training=True):
		""" init """
		self.args = args
		if not training:
			args.batch_size = 1
			args.seq_length = 1


		self.gpu_type = tf.float32
		
		print("Cell type is: ", args.model)

		if args.model == 'rnn':
			cell_fn = rnn.BasicRNNCell
		elif args.model == 'gru':
			cell_fn = rnn.GRUCell
		elif args.model == "basic_lstm":
			cell_fn = rnn.BasicLSTMCell
		elif args.model == 'lstm':
			cell_fn = tf.contrib.rnn.LSTMCell
		elif args.model == 'nas':
			cell_fn = rnn.NASCell
		else:
			raise Exception("model type not supported: {}".format(args.model))


		with tf.name_scope("Cells"):

			cells = []
			#for x in range(args.num_layers):
			#	cell = cell_fn(args.rnn_size)
			#	if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
			#		cell = rnn.DropoutWrapper(cell,
			#							  input_keep_prob=args.input_keep_prob,
			#							  output_keep_prob=args.output_keep_prob)
			#	cells.append(cell)
			## if odd number of layers > 1

			

			if args.num_layers == 3:
				print("\nHave three layers, making sandwich...")
				cells = []
				
				print("\tStarting rnn size: ", args.rnn_size)
				print("\tStarting seq_length: ", args.seq_length)

				outer_size = (args.rnn_size) 
				middle_size = outer_size // 2

				if training == True:
					args.rnn_size = outer_size
					print("\tChanged RNN size to: ", args.rnn_size)

				print("\tOuter size: {}  Middle size: {}".format(outer_size, 
					middle_size))

				cell_fn = rnn.LSTMCell
				
				## set up intersection stuff
				inter = tf.contrib.rnn.IntersectionRNNCell
				highway = tf.contrib.rnn.HighwayWrapper
				dropout_w = rnn.DropoutWrapper
				
				## project it onto the middle
				first = cell_fn(outer_size)#, num_proj = middle_size)
				middle = None
			
				## normalize the first layer before the highway
				avg_prob = abs(args.input_keep_prob / args.output_keep_prob)
				#first = rnn.LayerNormBasicLSTMCell(outer_size, dropout_keep_prob = avg_prob)

				use_highway = True

				if use_highway:
					args.using_highway = True
					middle = highway(cell_fn(outer_size))
				else:
					middle = cell_fn(outer_size)
					args.using_highway = False


				last = cell_fn(outer_size)				
		
				## dropout on first and last
				first = self.add_dropout(first, args.input_keep_prob, args.output_keep_prob)
				last = self.add_dropout(last, args.input_keep_prob, args.output_keep_prob)
				#first = dropout_w(first, input_keep_prob = args.input_keep_prob, 
				#		output_keep_prob=args.output_keep_prob)
				#last = dropout_w(last, input_keep_prob = args.input_keep_prob, 
				#		output_keep_prob=args.output_keep_prob)

				cells.append(first)
				cells.append(middle)
				cells.append(last)
				
				#out_prob = args.output_keep_prob
				#in_prob = args.input_keep_prob

				#if training and (in_prob < 1.0 or out_prob < 1.0):
				#	for i in range(3):
				#		cells[i] = rnn.DropoutWrapper(cells[i],
				#				input_keep_prob=in_prob,
				#				output_keep_prob = out_prob)

		

			print("Squishing {} cells into one".format(len(cells)))		
			print("Cells:  ", cells)
			self.cell = rnn.MultiRNNCell(cells, state_is_tuple=True)
			cell = self.cell
		#self.input_data = tf.placeholder(
		#	tf.int32, [args.batch_size, args.seq_length], name="step_input")
		#self.targets = tf.placeholder(
		#	tf.int32, [args.batch_size, args.seq_length], name="step_target")

		## all teh data for the epoch
		with tf.device("/gpu:0"):
			self.all_input_data = tf.Variable(tf.zeros([num_batches, args.batch_size, args.seq_length],
				dtype=self.gpu_type), dtype=self.gpu_type, trainable=False, name="all_inputs")

			self.all_target_data = tf.Variable(tf.zeros([num_batches, args.batch_size, args.seq_length],
				dtype=self.gpu_type), dtype=self.gpu_type, trainable=False, name="all_targets")

			self.step = tf.Variable(0, dtype=self.gpu_type, trainable=False, name="step")

			##data for each step
			self.input_data = tf.Variable(tf.zeros([args.batch_size, args.seq_length],
				dtype=self.gpu_type), dtype=self.gpu_type, name="batch_input", trainable=False)
			self.targets = tf.Variable(tf.zeros([args.batch_size, args.seq_length], 
				dtype=self.gpu_type), dtype=self.gpu_type, name="batch_targets", trainable=False)
		
		
		self.initial_state = cell.zero_state(args.batch_size, self.gpu_type)

		

		#self.all_data 

		self.num_batches = int(num_batches)

		## place holders for the data
		## we copy all the data for each epoch over
		## at the start to avoid excessive copies to the gpu
		##	"pin" the data to the gpu
		shape_tuple = (num_batches, args.batch_size, args.seq_length)
			
		with tf.device("/gpu:0"):
			self.step = tf.Variable(0, dtype=self.gpu_type, trainable=False, name="step")

			with tf.name_scope("inc_step"):
				self.inc_step = tf.assign_add(self.step, 1.0, name="inc_step")

		with tf.variable_scope('rnnlm'):
			#softmax_w = tf.get_variable("softmax_w",
			#							[args.rnn_size, args.vocab_size])
			
			## for new decoder
			softmax_w = tf.get_variable("softmax_w", 
					[args.batch_size, args.rnn_size, args.vocab_size])

			softmax_b = tf.get_variable("softmax_b", [args.vocab_size])

		## process input on cpu
		#with tf.device("/cpu:0"):
		with tf.name_scope("grab_step_data"):
			index = tf.to_int32(self.step, name="step_to_int")
			
			self.input_data = tf.assign(self.input_data, self.all_input_data[index])
			self.targets = tf.assign(self.targets, self.all_target_data[index])
			
			#self.input_data = self.all_input_data[index]
			#self.targets = self.all_target_data[index]
		
		#with tf.name_scope("inc_step"):
		#	self.inc_step = tf.assign_add(self.step, 1, name="inc_step") 

		
		## this maps vectors of len vocab_size => vectors of size rnn_size
		with tf.name_scope("get_embedding"):
			embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size], trainable=False)
			inputs = tf.nn.embedding_lookup(embedding, tf.to_int32(self.input_data))


		# dropout beta testing: double check which one should affect next line
		if training and args.output_keep_prob:
			inputs = tf.nn.dropout(inputs, args.output_keep_prob)
		## processing inputs on cpu
		#with tf.device("/cpu:0"):
		
		with tf.name_scope("flatten_input"):
			pass
			#print("inputs shape at start")
			#print(inputs.shape)
			#print("\nSplitting")
			#print("before: ", inputs.shape)
			#inputs = tf.split(inputs, args.seq_length, 1)
			#print("after split", np.array(inputs).shape)
			
			#inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
			#inputs = tf.squeeze(inputs)
			#def in_squeeze(item):
			#	return tf.squeeze(item, [1])
			
			#print("mapping...")
			#new_inputs = tf.map_fn(tf.squeeze, inputs)
			
			#print("after map: ", new_inputs.shape)
		
			#print(np.array(inputs).shape)
			#print("[0] shape: ", np.array(inputs[0]))
			#print("concatting")
			#inputs = tf.concat(inputs, axis=0)
			#print(inputs.shape)
	
		#def loop(prev, _):
		#	prev = tf.matmul(prev, softmax_w) + softmax_b
		#	prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
		#	return tf.nn.embedding_lookup(embedding, prev_symbol)

	

		with tf.name_scope("outputs"):
			#outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, 
			#		self.initial_state, cell, 
			#		loop_function=loop if not training else None, scope='rnnlm')
				
			print("\nBuilding decoder helper")
			self.seq_length = args.seq_length
			#output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])

			if not training:
				print("Using the inference helper")
				decoder_helper = s2s.ScheduledEmbeddingTrainingHelper(inputs, args.seq_length,
					embedding, 1.0)
			else:
				print("Using the training helper:")
				seq_lens = tf.fill([args.batch_size], args.seq_length)
				print("\tseq_lens: ", seq_lens.shape)
				print("\tinputs shape: ", inputs.shape)
				decoder_helper = s2s.TrainingHelper(inputs, seq_lens)
					
			## the meat
			decoder = s2s.BasicDecoder(self.cell, decoder_helper, self.initial_state)
	
			## what we want
			decoder_output, last_state, output_len = s2s.dynamic_decode(decoder)
			outputs = decoder_output.rnn_output

			#print("1: ", decoder_output)
			#print("\n\n2: ", last_state)
			#print("\n\n3: ", other)

			print("\n\noutputs: ", outputs.shape)
			
			#print("\nlast_state:", last_state)
			output = tf.to_float(outputs)
			print("Decoder outputs converted to floats")
			
			#print("The decoder stuff: ", len(stuff))
			#print(stuff[1])


		print("Getting logits")
		try:
			self.logits = tf.matmul(output, softmax_w) + softmax_b
		except Exception as ex:
			print("Error getting logits, bailing out: ", ex)
			exit(1)
		print("Got logits")
	
		## both of these are for sampling
		self.probs = tf.nn.softmax(self.logits)
		self.hardmax = s2s.hardmax(self.logits)

		tf.summary.scalar("max_prob", tf.reduce_max(self.probs))
		tf.summary.scalar("min_prob", tf.reduce_min(self.probs))
			
		print("Starting loss")
		print("logits: ", self.logits.shape)
		print("targets: ", self.targets.shape)
		#print("split: ", tf.split(self.logits, args.seq_length, 0))

		## make into [ batch_size, seq_len, vocab_size ] 
		split_logits = tf.reshape(self.logits, [args.batch_size, args.seq_length, args.vocab_size])

		loss_weights = tf.ones([args.batch_size, args.seq_length])

		loss = s2s.sequence_loss(split_logits, tf.to_int32(self.targets), loss_weights, name="compute_loss")

		
		tf.summary.scalar("max_loss", tf.reduce_max(loss))
		tf.summary.scalar("min_loss", tf.reduce_min(loss))

		with tf.name_scope('cost'):
			self.cost = loss #tf.reduce_sum(loss) / args.batch_size / args.seq_length
		self.final_state = last_state	
		self.lr = tf.Variable(0.0, trainable=False, name="lr")
		
		tf.summary.scalar("learning_rate", self.lr)

				
		## not used
		#gpus = self.args.gpu.split(",")
		#for i in range(len(gpus)):
		#	gpus[i] = "/gpu:" + gpus[i]
		#
		#print(gpus)
		

		with tf.name_scope("optimizer"):
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
		
		

		#grads, t_vars = zip(*self.optimizer.compute_gradients(self.cost))
		#grads, _ = tf.clip_by_global_norm(grads, self.args.grad_clip)
		#self.train_op = self.optimizer.apply_gradients(zip(grads, t_vars))

		with tf.name_scope("grad_clip"):
			gradients, variables = zip(*self.optimizer.compute_gradients(self.cost))
			#self.grads = gradients
			gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
			#self.clipped_grads = gradients
		with tf.name_scope("apply_grads"):
			self.train_op = self.optimizer.apply_gradients(zip(gradients, variables))
		
		with tf.name_scope("gradients"):
			if type(gradients) is list:
				for i in range(len(gradients)):
					print("Hanging grad histogram for: ", variables[i].name)
					tf.summary.histogram(variables[i].name, gradients[i])

		# instrument tensorboard
		tf.summary.histogram('logits', self.logits)
		tf.summary.histogram('loss', loss)
		tf.summary.scalar('train_loss', self.cost)


	def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
		print("In sample")
		
		state = sess.run(self.cell.zero_state(1, tf.float32))
		print("Set state")
		for char in prime[:-1]:
			x = np.zeros((1, 1))
			x[0, 0] = vocab[char]
			feed = {self.input_data: x, self.initial_state: state}
			[state] = sess.run([self.final_state], feed)
		
		print("Built primer")

		def weighted_pick(weights):
			t = np.cumsum(weights)
			s = np.sum(weights)
			return int(np.searchsorted(t, np.random.rand(1)*s))

		

		ret = prime
		char = prime[-1]
		for n in range(num):
			x = np.zeros((1, 1))
			x[0, 0] = vocab[char]
			feed = {self.input_data: x, self.initial_state: state}
			#print("Got feed: ", feed)
			
			#probs = sess.run(self.probs)
			#print("Got probs")
			#state = sess.run(self.final_state)
			#print("Got state")
			
			[probs, state] = sess.run([self.probs, self.final_state], feed)
			p = probs[0]

			#print("Got probs")
			if sampling_type == 0:
				sample = np.argmax(p)
			elif sampling_type == 2:
				if char == ' ':
					sample = weighted_pick(p)
				else:
					sample = np.argmax(p)
			else:  # sampling_type == 1 default:
				sample = weighted_pick(p)

			pred = chars[sample]
			ret += pred
			char = pred
		return ret

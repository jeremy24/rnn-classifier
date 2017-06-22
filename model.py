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

			## only working number of layers right now
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

		#with tf.variable_scope('rnnlm'):
		#	## for new decoder
		#	softmax_w = tf.get_variable("softmax_w", 
		#			[args.batch_size, args.rnn_size, args.vocab_size], trainable=False)
		#
		#	softmax_b = tf.get_variable("softmax_b", [args.vocab_size], trainable=False)

		## grad the batch data for the current step
		with tf.name_scope("grab_step_data"):
			index = tf.to_int32(self.step, name="step_to_int")
			
			self.input_data = tf.assign(self.input_data, self.all_input_data[index])
			self.targets = tf.assign(self.targets, self.all_target_data[index])
			
		
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
		
			if not training:
				print("Using the inference helper")
				start_tokens = tf.fill([args.seq_length], args.seq_length)
				end_token = 0
				print("\tState token: ", start_tokens.shape)
				print("\tEnd token: ", end_token)
				#decoder_helper = s2s.GreedyEmbeddingHelper(embedding, 
				#	start_tokens, end_token)
				seq_lens = tf.fill([args.batch_size], args.seq_length)
				embedded_inputs = tf.nn.embedding_lookup(embedding, tf.to_int32(self.input_data))
				decoder_helper = s2s.TrainingHelper(embedded_inputs, seq_lens)
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

			output = tf.to_float(outputs)
			print("Decoder outputs converted to floats")


		print("Getting logits")
		
	
		## the final layers
		##	maps the outputs  to [ vocab_size ] probs
		self.logits = tf.contrib.layers.fully_connected(output, args.vocab_size)

		
		## both of these are for sampling
		with tf.name_scope("probabilities"):
			print("Getting probs")
			self.probs = tf.nn.softmax(self.logits, name= "probs_softmax")
		
		with tf.name_scope("hardmax"):
			print("Getting hardmax")
			self.hardmax = tf.squeeze(s2s.hardmax(self.logits))

		with tf.name_scope("predict_index"):
			print("Probs shape: ", self.probs.shape)
			self.predict = tf.argmax(tf.squeeze(self.probs))



		tf.summary.scalar("max_prob", tf.reduce_max(self.probs))
		tf.summary.scalar("min_prob", tf.reduce_min(self.probs))
			

		## make into [ batch_size, seq_len, vocab_size ] 
		##	  it should already be this size, but this forces tf to recognize
		##	  the shape
		split_logits = tf.reshape(self.logits, 
				[args.batch_size, args.seq_length, args.vocab_size])


		with tf.name_scope("compute_loss"):
			loss_weights = tf.ones([args.batch_size, args.seq_length])
			loss = s2s.sequence_loss(split_logits, tf.to_int32(self.targets),
					loss_weights, name="compute_loss")

		
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
		
		print("Built feed dict")

		def weighted_pick(weights):
			t = np.cumsum(weights)
			s = np.sum(weights)
			return int(np.searchsorted(t, np.random.rand(1)*s))

		
		ret = prime
		char = prime[-1]

		print("Kicking off the predictions...")
		for n in range(num):
			x = np.zeros((1, 1))
			x[0, 0] = vocab[char]
			feed = {self.input_data: x, self.initial_state: state}
			
			
			#logits = sess.run([self.logits], feed)
			

			#probs, hardmax = sess.run([self.probs, self.hardmax], feed)			 
			predict = sess.run([self.predict], feed)

			#p = probs[0]

			#print("Got probs")
			#if sampling_type == 0:
			#	sample = np.argmax(p)
			#elif sampling_type == 2:
			#	if char == ' ':
			#		sample = weighted_pick(p)
			#	else:
			#		sample = np.argmax(p)
			#else:  # sampling_type == 1 default:
			#	sample = weighted_pick(p)
			
			sample = predict
		
			print("sample: ", sample)
			pred = chars[sample]
			ret += pred
			char = pred
		return ret

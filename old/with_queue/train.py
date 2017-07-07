from __future__ import print_function
import tensorflow as tf

import argparse
import time
import os
import json
import threading
from six.moves import cPickle

from utils import TextLoader
from model import Model




def main():
	parser = argparse.ArgumentParser(
						formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	
	parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
						help='data directory containing input.txt')
	
	parser.add_argument('--save_dir', type=str, default='save',
						help='directory to store checkpointed models')
	
	parser.add_argument('--log_dir', type=str, default='logs',
						help='directory to store tensorboard logs')
	
	parser.add_argument('--rnn_size', type=int, default=128,
						help='size of RNN hidden state')
	
	parser.add_argument('--num_layers', type=int, default=2,
						help='number of layers in the RNN')
	
	parser.add_argument('--model', type=str, default='lstm',
						help='rnn, gru, lstm, or nas')
	
	parser.add_argument('--batch_size', type=int, default=50,
						help='minibatch size')
	
	parser.add_argument('--seq_length', type=int, default=50,
						help='RNN sequence length')
	
	parser.add_argument('--num_epochs', type=int, default=50,
						help='number of epochs')
	
	parser.add_argument('--save_every', type=int, default=1000,
						help='save frequency')
	
	parser.add_argument('--grad_clip', type=float, default=5.,
						help='clip gradients at this value')
	
	parser.add_argument('--learning_rate', type=float, default=0.002,
						help='learning rate')
	
	parser.add_argument('--decay_rate', type=float, default=0.97,
						help='decay rate for rmsprop')
	
	parser.add_argument('--output_keep_prob', type=float, default=1.0,
						help='probability of keeping weights in the hidden layer')
	
	parser.add_argument('--input_keep_prob', type=float, default=1.0,
						help='probability of keeping weights in the input layer')
	
	parser.add_argument("--gpu", type=str, default="0", help="gpu[s] to run on")	
	
	parser.add_argument('--init_from', type=str, default=None,
						help="""continue training from saved model at this path. Path must contain files saved by previous training process:
							'config.pkl'		: configuration;
							'chars_vocab.pkl'	: vocabulary definitions;
							'checkpoint'		: paths to model file(s) (created by tf).
												  Note: this file contains absolute paths, be careful when moving files around;
							'model.ckpt-*'		: file(s) with model definition (created by tf)
						""")
	parser.add_argument("--print_cycle", type=int, default=1000, help="Cycle to print summary")
	args = parser.parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

	train(args)

## some aliases to shorten code
is_dir = os.path.isdir
is_file = os.path.isfile





def dump_args(args):
	filename = os.path.join(args.save_dir, "hyper_params.json")
	with open(filename, "w+") as fout:
		data = dict()
		args = vars(args)
		for key in args:
			data[key] = args[key]
		fout.write(json.dumps(data, sort_keys=True, indent=4, separators=(",", ":")))
		fout.close()








def train(args):
	data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length, args.num_epochs)
	args.vocab_size = data_loader.vocab_size
	
	dump_args(args)

	# check compatibility if training is continued from previously saved model
	if args.init_from is not None:
		# check if all necessary files exist
		assert is_dir(args.init_from)," %s must be a a path" % args.init_from
		assert is_file(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
		assert is_file(os.path.join(args.init_from,"chars_vocab.pkl")),"chars_vocab.pkl.pkl file does not exist in path %s" % args.init_from
		ckpt = tf.train.get_checkpoint_state(args.init_from)
		assert ckpt, "No checkpoint found"
		assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

		# open old config and check if models are compatible
		with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
			saved_model_args = cPickle.load(f)
		need_be_same = ["model", "rnn_size", "num_layers", "seq_length"]
		for checkme in need_be_same:
			assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme

		# open saved vocab/dict and check if vocabs/dicts are compatible
		with open(os.path.join(args.init_from, 'chars_vocab.pkl'), 'rb') as f:
			saved_chars, saved_vocab = cPickle.load(f)
		assert saved_chars==data_loader.chars, "Data and loaded model disagree on character set!"
		assert saved_vocab==data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"

	## check save dirs
	if not is_dir(args.save_dir):
		os.makedirs(args.save_dir)
	with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
		cPickle.dump(args, f)
	with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
		cPickle.dump((data_loader.chars, data_loader.vocab), f)

	

	

	print_cycle = args.print_cycle


	session_config = tf.ConfigProto(
			log_device_placement=False)

#	 print("Exiting before training...")
#	 exit(1)
	
	queue_cap = data_loader.num_batches * 4
	
	#batch_queue = tf.FIFOQueue(name="batch_queue", capacity=queue_cap,
	#		dtypes=[tf.int32, tf.int32], names=["x", "y"])
	
	batch_queue = tf.RandomShuffleQueue(name="random_queue", 
			capacity=queue_cap, min_after_dequeue=data_loader.num_batches,
			dtypes=[tf.int32, tf.int32], names=["x", "y"])

	print("Queue capacity is: ", queue_cap)
	
		

	def next_batch():
		return data_loader.queue_next_batch()
		#return load_queue(10)
	
	
	enqueue_op = batch_queue.enqueue(data_loader.queue_next_batch())

	## get a new thread coordinator
	runner = tf.train.QueueRunner(batch_queue, [enqueue_op] * 1)
	

		


	with tf.Session(config=session_config) as sess:
		

		print("kicking off queue runner")
		coord = tf.train.Coordinator()
		enqueue_threads = runner.create_threads(sess, coord, start=True)
		

		#sess.run(load_queue(data_loader.num_batches))	
		
		## the model builder uses dequeue, make sure it is built
		## after the queue runner is started
		print("Building model...")
		model = Model(args, batch_queue, training=True)
		print("Model built successfully")
		
		print("Initializing globals")
		sess.run(tf.global_variables_initializer())

		print("Saving globals")
		saver = tf.train.Saver(tf.global_variables())
		# restore model
		if args.init_from is not None:
			saver.restore(sess, ckpt.model_checkpoint_path)
	
		print("Hanging summary writer")
		summaries = tf.summary.merge_all()
		writer = tf.summary.FileWriter(
			os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))
		
		writer.add_graph(sess.graph)	
		
		total_steps = args.num_epochs * data_loader.num_batches

		print("num_batches", data_loader.num_batches)
		print("Starting...")
		big_start = time.time()
			
		for epoch in range(args.num_epochs):
			sess.run(tf.assign(
				model.lr, 
				args.learning_rate * (args.decay_rate ** epoch)))
			
			#data_loader.reset_batch_pointer()
			print("Setting initial state")
			state = sess.run(model.initial_state)

			#load_queue(data_loader.num_batches)		

			batch_start = time.time()
			start = time.time()
			for batch in range(data_loader.num_batches):
				
				print("batch: ", batch, " queue size: ", batch_queue.size().eval())				
				print("time since last batch: ", time.time() - start)
				x, y, left = data_loader.next_batch()
				feed = {model.input_data: x, model.targets: y}
				for i, (c, h) in enumerate(model.initial_state):
					feed[c] = state[i].c
					feed[h] = state[i].h
				
				#print("sess run checker")
				#print(model.cost is not  None)
				#print(model.final_state is not None)	
				#print(model.train_op is not None)
				#print(summaries is not None)
				#print("checker done: ")
				train_loss, state, _ = sess.run(
					[model.cost, model.final_state, model.train_op], feed)

				
				step = epoch * data_loader.num_batches + batch
				
				# instrument for tensorboard
				if step % 50 == 0:
						
					summ, train_loss, state, _ = sess.run([summaries, model.cost, model.final_state, model.train_op])
					writer.add_summary(summ, epoch * data_loader.num_batches + batch)
				elif False: ## skip this
					feed = dict()
					start = time.time()
					train_loss = sess.run(model.cost, feed)
					print("cost took: ", time.time() - start)
					start = time.time()
					state = sess.run(model.final_state, feed)
					print("state took: ", time.time() - start)
					start = time.time()
					sess.run(model.train_op, feed)
					print("train took: ", time.time() - start)
					#train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op])

			   

				if batch % print_cycle == 0 and batch > 0:
					end = time.time()
					tf.summary.scalar("time_per_1k", end-start)
					print("{}/{} (epoch {}), train_loss = {:.3f}, time/{} = {:.3f}"
						.format(step, total_steps, epoch, train_loss, print_cycle, end - batch_start))

	
				if (step) % args.save_every == 0:
					# save for the last result
					checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
					saver.save(sess, checkpoint_path, global_step = step)
					print("model saved to {}".format(checkpoint_path))
				start = time.time()
			big_end = time.time()
			print("Total time: ", big_end - big_start)
		saver.save(sess, checkpoint_path, global_step = total_steps)


if __name__ == '__main__':
	main()

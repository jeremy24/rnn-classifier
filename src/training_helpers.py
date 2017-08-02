import numpy as np
import tensorflow as tf


class DumpIntoNamespace:
	"""
	Convert a dict into a NameSpace
	"""

	def __init__(self, env, *args):
		self.vars = dict([(x, env[x]) for v in args for x in env if v is env[x]])

	def __getattr__(self, name):
		print(self.vars)
		return self.vars[name]


class Confusion(object):
	def __init__(self, sess, model, feed):
		self.sess = sess
		self.model = model
		self.feed = feed

	def __enter__(self):
		return self.sess.run(self.model.confusion, self.feed)

	def __exit__(self, error_type, value, trace):
		if value:
			print("Confusion Error: {}\n{}\n".format(error_type, value))
			exit(1)


class NormalTrain(object):
	def __init__(self, sess, model, feed):
		self.sess = sess
		self.feed = feed
		self.args = [model.cost, model.final_state, model.train_op, model.loss]

	def __enter__(self):
		cost, state, _, loss = self.sess.run(self.args, self.feed)
		return state, loss

	def __exit__(self, error_type, value, trace):
		if value:
			print("NormalTrain Error: {}\n{}\n{}".format(error_type, value, trace))
			exit(1)


class PrintTrain(object):
	def __init__(self, sess, model, summaries, feed):
		self.sess = sess
		self.feed = feed
		self.args = [summaries, model.loss, model.final_state,
					 model.train_op, model.lr, model.global_step, model.their_confusion]

	def __enter__(self):
		summary, loss, state, _, lr, g_step, confusion = self.sess.run(self.args, feed_dict=self.feed)
		return {"summary": summary, "train_loss": loss,
				"state": state, "lr": lr, "g_step": g_step, "confusion": confusion}

	def __exit__(self, error_type, value, trace):
		if value:
			print("PrintTrain Error: {}\n{}\n".format(error_type, value))
			exit(1)


def calculate_patience(epoch_loss, patience, lowest_epoch_loss):
	# if new epoch loss is 0.5% lower than lowest loss, extend patience
	this_epoch_loss = np.median(epoch_loss)
	if this_epoch_loss + (this_epoch_loss * .008) <= lowest_epoch_loss:
		patience += 2
		lowest_epoch_loss = this_epoch_loss
		print("Added 2 to patience, new lowest loss is {:.5f}".format(lowest_epoch_loss))
	elif this_epoch_loss + (this_epoch_loss * .005) <= lowest_epoch_loss:
		patience += 1
		lowest_epoch_loss = this_epoch_loss
		print("Added 1 to patience, new lowest loss is {:.5f}".format(lowest_epoch_loss))
	elif this_epoch_loss + (this_epoch_loss * .008) > lowest_epoch_loss:
		patience -= 2
		print("Removing 2 from patience")
	elif this_epoch_loss + (this_epoch_loss * .005) > lowest_epoch_loss:
		patience -= 1
		print("Removing 1 from patience")
	return patience, lowest_epoch_loss


def get_sess_config():
	# used if you want a lot of logging
	sess_config = tf.ConfigProto()

	# used to watch gpu memory thats actually used
	sess_config.gpu_options.allow_growth = True

	# used to show where things are being placed
	sess_config.log_device_placement = False

	jit_level = tf.OptimizerOptions.ON_1

	sess_config.graph_options.optimizer_options.global_jit_level = jit_level
	return sess_config


def get_metrics_holder():
	data = dict()
	data["losses"] = list()
	data["avg_time_per_step"] = list()
	data["logged_time"] = list()
	data["false_positives"] = list()
	data["false_negatives"] = list()
	data["step"] = list()
	data["label_ratio"] = list()
	return data


def copy_data_info(args, data_loader):
	args.vocab_size = data_loader.vocab_size
	args.batch_size = data_loader.batch_size
	args.label_ratio = data_loader.ratio
	args.num_classes = data_loader.num_classes
	args.num_batches = data_loader.num_batches
	args.num_chars = data_loader.num_chars
	return args


def get_flags(num_batches, num_epochs, epoch, batch, step):
	last_batch = batch == num_batches - 1
	last_epoch = epoch == num_epochs - 1
	s = step - (epoch * num_batches)
	last_in_epoch = s == num_batches - 1
	return last_batch, last_epoch, last_in_epoch

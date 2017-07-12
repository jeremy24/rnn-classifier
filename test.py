""" test a tensorflow model """

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import os

from six.moves import cPickle

import numpy as np
import tensorflow as tf
import sklearn.metrics as skmetrics

# my stuff
from model import Model
from data_loader import TextLoader


def main():
	"""main stuff"""
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--save_dir', type=str, default='save',
						help='model directory to load from')
	parser.add_argument('-n', type=int, default=100,
						help='number of test batches to sample')
	parser.add_argument('--sample', type=int, default=1,
						help='0 to use max at each timestep, 1 to sample at '
							 'each timestep, 2 to sample on spaces')

	args = parser.parse_args()
	test(args)


def make_html(original, expected, labels):
	if len(original) != len(labels):
		raise ValueError("label and original lengths don't match: {} != {}".format(len(original), len(labels)))

	tp = 100 * np.sum(np.logical_and(expected, labels)) / len(labels)
	fp = 100 * np.sum(np.logical_and(np.logical_not(expected), labels)) / len(labels)

	tn = 100 * np.sum(np.logical_and(np.logical_not(expected), np.logical_not(labels))) / len(labels)
	fn = 100 * np.sum(np.logical_and(expected, np.logical_not(labels))) / len(labels)

	assert int(tp + fp + tn + fn) == 100

	html = "<p>"

	html += "<p><strong>True Positives:   {:.2f}% </strong></p>".format(tp)
	html += "<p><strong>False Positives:  {:.2f}% </strong></p>".format(fp)
	html += "<p><strong>True Negatives:   {:.2f}% </strong></p>".format(tn)
	html += "<p><strong>False Negatives:  {:.2f}% </strong></p>".format(fn)

	def bold(item, char_color="black"):
		return "<strong  style='color: {}' >{}</strong>".format(char_color, item)

	# print the labeling we got
	html += "<p><strong>We Got: </strong>"
	for i in range(len(labels)):
		char = original[i]
		label = labels[i]
		wanted = expected[i]
		if char == " " and (label or wanted):
			char = "_"
		if char == "\n" and (label or wanted):
			char = "(N)"
		if char == "\t" and (label or wanted):
			char = "(T)"
		color = "black"
		do_bold = True
		if wanted == 1 and label == 1:
			color = "blue"
		if wanted == 1 and label == 0:
			color = "red"
		if wanted == 0 and label == 1:
			color = "orange"
		if wanted == 0 and label == 0:
			do_bold = False

		html += bold(char, color) if do_bold else char
	html += "</p>"

	# print the expected labeling
	html += "<p><strong>Wanted: </strong>"
	for i in range(len(labels)):
		char = original[i]
		wanted = expected[i]
		label = labels[i]
		if char == " " and (wanted or label):
			char = "_"
		if char == "\n" and (wanted or label):
			char = "(N)"
		if char == "\t" and (wanted or label):
			char = "(T)"
		wanted = expected[i]
		do_bold = False
		if wanted == 1:
			do_bold = True
			color = "blue"
		html += bold(char, color) if do_bold else char
	html += "</p>"

	html += "<p>"
	for i in range(len(labels)):
		label = labels[i]
		html += " " + str(label)
	html += "</p>"

	html += "<p>"
	for i in range(len(labels)):
		wanted = expected[i]
		html += " " + str(wanted)
	html += "</p>"

	html += "<p>"
	return html


def get_chars(x_batch, flatten=True):
	orig = [list() for x in x_batch]
	i = 0
	for sub in x_batch:
		for idx in sub:
			orig[i].append(chr(idx))
		i += 1
	if flatten:
		return np.array(orig).flatten().tolist()
	return orig


def run_test(sess, model, x_seq, y_seq, args, state, number=0):
	""" run a test for a single batch """
	# both are => [ batch_size, seq_length ]
	# x_seq = batch[0]
	# y_seq = batch[1]

	try:
		# print("x_seq shape: ", x_seq.shape, "y_seq shape: ", y_seq.shape)

		feed = {model.input_data: x_seq, model.targets: y_seq}  # ,  model.initial_state: state}
		probs, state, loss = sess.run([model.probs, model.final_state, model.cost], feed)

		y_seq_ = probs

		# print("y_seq_ shape: ", y_seq_.shape)

		original = get_chars(x_batch=x_seq, flatten=False)
		y_bar = list()

		for item in y_seq_:
			sublist = list()
			for sub in item:
				sublist.append(np.argmax(sub))
			y_bar.append(sublist)

		y_bar = np.array(y_bar)
		# print("y_bar: ", y_bar[0])
		# print("y_seq: ", y_seq[0])
		# accuracy = np.mean(np.equal(y_seq, y_bar))

		html = "<html><body><div>"
		html += "<h3>Spaces in the sequences are replaced with an underscore if they " \
				"were labeled or expected to be labeled</h3>"
		html += "<h4>Blue: labeled and wanted label</h4>"
		html += "<h4>Red: not labeled and wanted label</h4>"
		html += "<h4>Orange: labeled and wanted no label</h4>"
		html += "<div><p>Ratio of NO labels to YES labels is {:.2f}:1</p>".format(args.label_ratio)
		html += "<p> {:.5f}% of the characters are labeled YES".format(1.0 / args.label_ratio * 100.0)
		html += "<p>This model was trained for {:,} epoch[s] with a batch size of {:,}".format(
			args.num_epochs, args.batch_size
		)
		html += "<p>Decay Rate: {:.2f} </p><p> RNN Size: {:,}</p><p>Vocab Size: {:,}</p>".format(
			args.decay_rate, args.rnn_size, args.vocab_size
		)
		html += "<p>Out Keep P: {:.2f}</p><p>In Keep P: {:.2f}</p><p>LR: {:.5f}</p><p>Num Layers: {:,}</p>".format(
			args.output_keep_prob, args.input_keep_prob, args.learning_rate, args.num_layers
		)
		html += "<p>Model: {}</p><p>Number of Params: {:,}</p><p>Sequence length: {:,}</p>".format(
			args.model, args.num_params, model.seq_length
		)
		html += "</div><div><ol>"
		for i in range(len(original)):
			html += "<li> {} </li>".format(make_html(original[i], y_seq[i], y_bar[i]))

		html += "</ol></div></body></html>"

		try:
			y_seq = np.ndarray.flatten(y_seq)
			y_bar = np.ndarray.flatten(y_bar)
			with open("labeled.html", "w") as fout:
				fout.write(html)
			confusion = skmetrics.confusion_matrix(y_seq, y_bar)
		except ValueError as ex:
			print("Error build confusion:", ex)
			print(y_seq.shape)
			print("labels y_bar:", set(y_bar))
			print("labels y_seq:", set(y_seq))
			exit(1)

		tn = confusion[0, 0]
		fp = confusion[0, 1]
		fn = confusion[1, 0]
		tp = confusion[1, 1]

		precision = tp / (fp + tp)
		recall = tp / (tp + fn)
		accuracy = (tn + tp) / (tn + fp + fn + tp)
		sensitivity = recall  # same thing
		specificity = tn / (tn + fp)

		ret = dict()
		ret["accuracy"] = accuracy
		# ret["state"] = state
		ret["loss"] = loss
		ret["precision"] = precision
		ret["recall"] = recall
		ret["sensitivity"] = sensitivity
		ret["specificity"] = specificity

		# print(tp, fp, tn, fn)
		# print(tp, "/", "(", fp, "+", tp, ")")
		# print(ret)

		return ret

	except Exception as ex:
		print("run_test: ", ex)
		exit(1)


def test(args):
	""" run the tests """
	saved_args = None
	data_loader = None
	model = None

	with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as fin:
		saved_args = cPickle.load(fin)

	print("Saved args: ", saved_args)

	data_loader = TextLoader(saved_args.data_dir, saved_args.save_dir, saved_args.batch_size,
							 saved_args.seq_length)

	data_loader.batches = data_loader.test_batches
	data_loader.num_batches = len(data_loader.batches)

	print("Data loaded in")

	args.vocab_size = data_loader.vocab_size

	model = Model(saved_args, 1, training=False)

	print("Sample model built")

	os.environ["CUDA_VISIBLE_DEVICES"] = "0"

	def t_print(stuff):
		"""pretty print stats from a list"""
		print("\tAvg: {:.3f}".format(np.mean(stuff)))
		print("\tMax: {:.3f}".format(np.max(stuff)))
		print("\tMin: {:.3f}".format(np.min(stuff)))
		print("\tStd: {:.3f}".format(np.std(stuff)))

	with tf.Session() as sess:
		tf.global_variables_initializer().run()

		saver = tf.train.Saver(tf.global_variables())
		print("Got saver foir trainable variables")

		ckpt = tf.train.get_checkpoint_state(args.save_dir)
		print("Loaded checkpoint from save dir")
		if ckpt and ckpt.model_checkpoint_path:
			print("Restoring...")
			saver.restore(sess, ckpt.model_checkpoint_path)
			print("Restored checkpoint successfully")

			# state = sess.run(model.cell.zero_state(saved_args.batch_size, tf.float32))

			print("\ngot initial state")
			print("\nGrabbed batches")
			print("Running", args.n, "test batches")

			losses = list()
			accs = list()
			precs = list()
			recalls = list()
			i = 0

			for batch in data_loader.test_batches:
				# state = sess.run(model.cell.zero_state(saved_args.batch_size, tf.float32))
				if i == args.n:  # number to run
					break
				# batch => [2]	   [x, y] pairs of data where
				# x and y are => [ batch_size, seq_length ]
				if i % 100 == 0:
					print("On batch:", i)

				# accuracy, _, loss = run_test(sess, model, batch[0], batch[1], state)
				x = batch[0]
				y = batch[1]
				metrics = run_test(sess, model, x, y, saved_args, None)

				losses.append(metrics["loss"])
				accs.append(metrics["accuracy"])
				precs.append(metrics["precision"])
				recalls.append(metrics["recall"])
				i += 1
				break

			print("\nFor {} batches".format(i))
			print("Accuracy:")
			t_print(accs)

			print("Loss:")
			t_print(losses)

			print("Precision:")
			t_print(precs)

			print("Recall:")
			t_print(recalls)


if __name__ == '__main__':
	main()

""" test a tensorflow model """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import matplotlib
import plotly.plotly as py
import plotly.graph_objs as go

matplotlib.use("Agg")


import jinja2 as jinja
import numpy as np
import sklearn.metrics as skmetrics
import tensorflow as tf
from six.moves import cPickle
from matplotlib import pyplot as plt

from data_loader import TextLoader
from model import Model



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


def make_hist(x, y, save_location, label_max, title,
			  nbins=16, step=100, filetype="png", y_label="Position in Sequence",
			  x_label="Sequence Number"):
	print("\nPlotting")
	print("\tx length: ", len(x))
	print("\ty length: ", len(y))
	plt.ioff()
	plt.hist2d(x, y, bins=nbins)
	plt.title(title)
	plt.ylabel(y_label)
	plt.xlabel(x_label)
	plt.xticks(range(0, label_max, step), range(0, label_max, step))
	plt.savefig(save_location + str(title).replace(" ", "_") + "." + filetype)


def make_lineplot(x, y, title, save_location, xlabel="x", ylabel="y",
				  filetype="png", label_max=25, step=5):
	x = np.array(x).flatten()

	print("\nLineplot:")
	print("\tTitle: ", title)
	plt.ioff()
	if y is None:
		plt.plot(x)
	else:
		y = np.array(y).flatten()
		print(x)
		print(y)
		plt.plot(x, y)

	plt.title(str(title))
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.xticks(range(0, label_max, step), range(0, label_max, step))
	plt.savefig(save_location + str(title).replace(" ", "_") + "." + filetype)

def label_compare(have, want, save_location, nbins=16, step=50):
	want_coord = [[], []]
	have_coord = [[], []]
	i = 0
	have = np.array(have)
	want = np.array(want)
	for h, w in zip(have, want):
		j = 0

		for h_num, w_num in zip(h, w):
			if h_num == 1:
				have_coord[0].append(i)
				have_coord[1].append(j)
			if w_num == 1:
				want_coord[0].append(i)
				want_coord[1].append(j)
			j += 1
		i += 1
	make_hist(want_coord[0], want_coord[1], save_location, len(want),
			  "Labels Wanted", nbins=nbins, step=step)
	make_hist(have_coord[0], have_coord[1], save_location, len(want),
			  "Labels Got", nbins=nbins, step=step)


def right_wrong(have, want, save_location, nbins=16, step=50):
	fp = np.logical_and(have.astype("bool"), np.logical_not(want.astype("bool")))

	fn = np.logical_and(np.logical_not(have), want)

	fp_fn = np.logical_or(fp, fn)
	fp_fn_coords = [[], []]

	tp = np.logical_and(have.astype("bool"), want.astype("bool"))
	tn = np.logical_and(np.logical_not(have.astype("bool")), np.logical_not(want.astype("bool")))

	tp_tn = np.logical_or(tp, tn)
	tp_tn_coords = [[], []]

	i = 0
	for bad, good in zip(fp_fn, tp_tn):
		j = 0
		for g, b in zip(good, bad):
			if b:
				fp_fn_coords[0].append(i)
				fp_fn_coords[1].append(j)
			if g:
				tp_tn_coords[0].append(i)
				tp_tn_coords[1].append(j)

			j += 1
		i += 1

	make_hist(fp_fn_coords[0], fp_fn_coords[1], save_location, len(want),
			  "Labels Wrong", nbins=nbins, step=step)
	make_hist(tp_tn_coords[0], tp_tn_coords[1], save_location, len(want),
			  "Labels correct", nbins=nbins, step=step)


def do_vis(nbins=16, todo=50, step_size=50):
	results = np.load("./models/common_words/save/results.npy")
	wanted =  np.load("./models/common_words/save/wanted.npy")

	have = np.reshape(results, [-1, results.shape[2]])[:todo]
	want = np.reshape(wanted, [-1, results.shape[2]])[:todo]

	save_location = "./figures/"

	label_compare(have, want, save_location, nbins, step=step_size)
	right_wrong(have, want, save_location, nbins, step=step_size)

def make_html(original, expected, labels):
	if len(original) != len(labels):
		raise ValueError("label and original lengths don't match: {} != {}".format(len(original), len(labels)))

	tp = 100 * np.sum(np.logical_and(expected, labels)) / len(labels)
	fp = 100 * np.sum(np.logical_and(np.logical_not(expected), labels)) / len(labels)

	tn = 100 * np.sum(np.logical_and(np.logical_not(expected), np.logical_not(labels))) / len(labels)
	fn = 100 * np.sum(np.logical_and(expected, np.logical_not(labels))) / len(labels)

	assert int(tp + fp + tn + fn) == 100

	html = "<p>"

	header = None
	with open("./partials/section_header.html") as fin:
		header = jinja.Template(fin.read())
		render = header.render(fn=fn, fp=fp, tn=tn, tp=tp)
		html += render

	# html += "<p><strong>True Positives:   {:.2f}% </strong></p>".format(tp)
	# html += "<p><strong>False Positives:  {:.2f}% </strong></p>".format(fp)
	# html += "<p><strong>True Negatives:   {:.2f}% </strong></p>".format(tn)
	# html += "<p><strong>False Negatives:  {:.2f}% </strong></p>".format(fn)

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

	# html += "<p>"
	# for i in range(len(labels)):
	# 	label = labels[i]
	# 	html += " " + str(label)
	# html += "</p>"
	#
	# html += "<p>"
	# for i in range(len(labels)):
	# 	wanted = expected[i]
	# 	html += " " + str(wanted)
	# html += "</p>"

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
		# print("y_seq: ", y_seq[0])n
		# accuracy = np.mean(np.equal(y_seq, y_bar))


		params = {
			"label_ratio": round(args.label_ratio, 3),
			"label_ratio_percent": round(1.0 / args.label_ratio * 100.0, 3),
			"num_epochs": args.num_epochs,
			"batch_size": args.batch_size,
			"decay_rate": args.decay_rate,
			"rnn_size": args.rnn_size,
			"vocab_size": args.vocab_size,
			"out_prob": args.output_keep_prob,
			"in_prob": args.input_keep_prob,
			"lr": args.learning_rate,
			"num_layers": args.num_layers,
			"model_type": args.model,
			"num_params": args.num_params,
			"seq_length": model.seq_length,
			"item_list": [make_html(original[i], y_seq[i], y_bar[i]) for i in range(len(original))]
		}


		# print("Built params")

		with open("./partials/header.html") as fin:
			head = fin.read()

			try:
				head = jinja.Template(head)
			except Exception as ex:
				print("Error compiling head template", ex)
				exit(1)
			head = head.render(params)




		html = head

		# y = y_seq[0]
		# y_ = y_bar[0]
		# horiz = [ord(c) for c in y_seq]

		result = y_bar

		#
		# html += "</div><div><ol>"
		# for i in range(len(original)):
		# 	html += "<li> {} </li>".format(make_html(original[i], y_seq[i], y_bar[i]))
		#
		# html += "</ol></div></body></html>"

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

		return ret, result

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
		print("Got saver for trainable variables")

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

			results = list()
			wanted = list()

			for batch in data_loader.test_batches:
				# state = sess.run(model.cell.zero_state(saved_args.batch_size, tf.float32))
				if i == args.n:  # number to run
					break
				# batch => [2]	   [x, y] pairs of data where
				# x and y are => [ batch_size, seq_length ]

				print("On batch:", i)

				# accuracy, _, loss = run_test(sess, model, batch[0], batch[1], state)
				x = batch[0]
				y = batch[1]
				metrics, y_bar = run_test(sess, model, x, y, saved_args, None)

				results.append(y_bar)
				wanted.append(y)
				print("Results len: ", len(results))

				losses.append(metrics["loss"])
				accs.append(metrics["accuracy"])
				precs.append(metrics["precision"])
				recalls.append(metrics["recall"])
				i += 1

			y_bar = np.array(results)
			wanted = np.array(wanted)

			print("y_bar shape: ", y_bar.shape)
			assert len(results) == i
			assert len(wanted) == len(results)

			np.save(os.path.join(args.save_dir, "results.npy"), y_bar)
			np.save(os.path.join(args.save_dir, "wanted.npy"), wanted)

			make_lineplot(range(len(losses)), losses, "Loss",
						  "./figures/", xlabel="Sequence Number",
						  ylabel="Loss", label_max=i, step=1)

			make_lineplot(range(len(accs)), accs, "Accuracy",
						  "./figures/", xlabel="Sequence Number",
						  ylabel="Accuracy", label_max=i, step=1)



			do_vis(nbins=saved_args.seq_length // 3, todo=50)


			print("\nFor {} batches".format(i))
			print("Accuracy:")
			t_print(accs)

			print("Loss:")
			t_print(losses)

			print("Precision:")
			t_print(precs)

			print("Recall:")
			t_print(recalls)
		else:
			print("\nInvalid checkpoint file")


if __name__ == '__main__':
	main()

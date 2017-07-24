from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import os
import sys
import glob


REPLACE_CHAR = "#"


def replace(string, start, end, inclusive=True):
	orig_len = len(string)
	if inclusive:
		end += 1

	for i in range(start, end):
		string[i] = REPLACE_CHAR

	assert orig_len == len(string), "{} != {}".format(orig_len, len(string))
	return string


def get_start_ends(ann, filter_anns=None):
	ann = [str(item).strip().split() for item in ann]
	ann = [item[1:] for item in ann if "T" in item[0]]
	ann = [item[1:] for item in ann if filter_anns is None or item[0] in filter_anns]
	ret = list()
	for row in ann:
		idxs = list()

		if ";" in "".join(row):
			semis = "".join(row).count(";")
			todo = (semis * 2)
			items = list()
			for item in row:
				if ";" in item:
					items += item.split(";")
				else:
					items.append(item)
			i = 0
			while i < todo:
				idxs.append(int(items[i]))
				i += 1
		else:
			idxs.append(int(row[0]))
			idxs.append(int(row[1]))

		assert len(idxs) % 2 == 0, "Length of idxs is not even"
		for i in range(0, len(idxs), 2):
			ret.append((int(idxs[i]), int(idxs[i+1])))
	for (start, end) in ret:
		assert start < end, "{} >= {}".format(start, end)
	return ret


def pair_files(dirname, anns=None, filename=None):
	print("\nPairing files")
	print("\tDirname: {}".format(dirname))
	assert os.path.exists(dirname), "provided dir {} does not exist".format(dirname)

	if filename is None:
		text_path = dirname + "/**/*.txt"
		print("\tText path: {}".format(text_path))
		text_files = glob.glob(text_path, recursive=True)
	else:
		text_files = [os.path.join(dirname, filename)]
	# text_files = ["/data/forJeremy/seed_ann.6.29.2017/seed198-rama/288387900092.txt"]
	ann_files = list()

	for file in text_files:
		file = str(file).replace(".txt", ".ann")
		ann_files.append(file)

	assert len(ann_files) == len(text_files), "number of .ann and .txt does not match"

	print("\tHave {:,} pairs of files".format(len(text_files)))

	clean_data = list()

	for (text_file, ann_file) in zip(text_files, ann_files):
		with open(text_file, "r") as text:
			text_data = "".join(text)
		with open(ann_file, "r") as ann:
			pairs = get_start_ends(ann, anns)
			orig_len = len(text_data)
			text_data = list(text_data)
			for (start, end) in pairs:
				text_data = replace(text_data, start, end)
			text_data = "".join(text_data)
			assert orig_len == len(text_data), "{} != {}".format(orig_len, len(text_data))

		clean_data.append((str(text_file), text_data))

	return clean_data


def write_out(base_path, clean_data, ext=".labeled", folder="labeled"):
	for (name, data) in clean_data:
		name = name.split("/")
		filename = name[-1]
		name[-1] = folder
		name = "/".join(name)

		# make dir if not exists
		if not os.path.exists(name):
			os.makedirs(name)
		name = os.path.join(name, filename)
		name += ext
		with open(name, "w") as fout:
			fout.write(data)


def process_ann_files(dirpath, replace_char, ext=".labeled", folder="labeled"):
	try:
		anns = None
		assert len(replace_char) == 1, "Replacement char must be of length 1"
		global REPLACE_CHAR
		REPLACE_CHAR = replace_char
		data = pair_files(dirpath, anns)
		write_out(dirpath, data, ext=ext, folder=folder)
	except Exception as ex:
		print("Error processing ann files: ", ex)
		exit(1)

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
		have_done = 0

		if ";" in "".join(row):
			print(row)
			semis = "".join(row).count(";")
			print("have {} semis".format(semis))
			todo = (semis * 2)
			items = list()
			for item in row:
				if ";" in item:
					items += item.split(";")
				else:
					items.append(item)
			print("todo", todo)
			print(items)
			while todo > 0:
				print(todo)
				idxs.append(int(items[i]))
				print("appending")
				todo -= 1
		else:
			idxs.append(int(row[0]))
			idxs.append(int(row[1]))

		for i in range(0, len(idxs), 2):
			ret.append((int(idxs[i]), int(idxs[i+1])))
	for (start, end) in ret:
		assert start < end, "{} >= {}".format(start, end)
	return ret

def pair_files(dirname, anns):
	print("\nPairing files")
	print("\tDirname: {}".format(dirname))
	assert os.path.exists(dirname), "provided dir {} does not exist".format(dirname)
	text_path = os.path.join(dirname, "*.txt")

	print("\tText path: {}".format(text_path))

	text_files = glob.glob(text_path)
	# text_files = ["/data/forJeremy/seed_ann.6.29.2017/seed198-rama/288387900092.txt"]
	ann_files = list()

	for file in text_files:
		file = str(file).replace(".txt", ".ann")
		ann_files.append(file)

	assert len(ann_files) == len(text_files), "number of .ann and .txt does not match"

	print("\tHave {:,} pairs of files".format(len(text_files)))

	clean_data = list()

	for (text_file, ann_file) in zip(text_files, ann_files):
		replaced = 0
		with open(text_file, "r") as text:
			text_data = "".join(text)
		with open(ann_file, "r") as ann:
			print("\n\t", ann_file)
			pairs = get_start_ends(ann, anns)
			orig_len = len(text_data)
			text_data = list(text_data)
			for (start, end) in pairs:
				text_data = replace(text_data, start, end)
			text_data = "".join(text_data)
			assert orig_len == len(text_data), "{} != {}".format(orig_len, len(text_data))

		clean_data.append((str(text_file).split("/")[-1], text_data))

	print("\tClean data length: {}".format(len(clean_data)))
	return clean_data


def write_out(base_path, clean_data):
	print("\nWriting out labeled data")
	clean_dir = os.path.join(base_path, "labeled/")
	if os.path.exists(clean_dir):
		for file in os.listdir(clean_dir):
			os.remove(os.path.join(clean_dir, file))
		os.removedirs(clean_dir)
	os.makedirs(clean_dir)
	assert os.path.exists(clean_dir)
	print("\nClean dir: {}".format(clean_dir))
	for (name, data) in clean_data:
		path = os.path.join(clean_dir, name + ".labeled")
		print("\tPath: {}".format(path))
		print("\tName: {}".format(name))
		with open(path, "w") as fout:
			fout.write(data)
	print("Done writing out labeled data")


def main():
	anns = None
	print(sys.argv)
	data = pair_files(sys.argv[1], anns)
	write_out(sys.argv[1], data)



if __name__ == "__main__":
	main()

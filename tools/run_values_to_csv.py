import json
import csv
import os
import sys




def load_json(path):

	with open(path, "r") as fin:
		data = json.load(fin)
		assert len(data) > 0, "No data from json file"
		assert data["data"]
		metrics = dict(data["data"])

		length = len(metrics["false_positives"])
		assert len(metrics["false_negatives"]) == length, "false_negatives length does not match"
		assert len(metrics["label_ratio"]) == length, "label_ratio length does not match"
		assert len(metrics["losses"]) == length, "losses length does not match"
		assert len(metrics["avg_time_per_step"]) == length, "avg_time_per_step length does not match"

	keys = ["false_positives", "false_negatives",
				"label_ratio", "losses", "avg_time_per_step"]

	no_step = True
	step = None
	if "step" in metrics.keys():
		keys.append("step")
		no_step = False
	else:
		step = int(data["print_cycle"])


	# remove keys we don't want
	for key in list(metrics.keys()):
		if key not in keys:
			metrics.pop(key, None)

	# make dict of lists into list of dicts
	ret = list()
	for i in range(length):
		obj = dict()
		for key in keys:
			obj[key] = metrics[key][i]
		if no_step:
			obj["step"] = (i + 1) * step
		ret.append(obj)

	return ret


def write_csv(data, dir_path):
	dir_path = str(dir_path)
	split = dir_path.split("/")[0:-1]
	if split[-1] != "/":
		split.append("/")
	print(dir_path.split("/")[0:-2])
	dir_path = "/".join(dir_path.split("/")[0:-2])
	path = os.path.join(dir_path, "data_by_step.csv")
	assert type(data) == list, "Data must be a list"
	assert len(data) > 0
	assert type(data[0]) == dict, "Data must be a list of dicts"

	with open(path, "w") as fout:
		names = list(data[0].keys())
		writer = csv.DictWriter(fout, fieldnames=names)
		writer.writeheader()
		writer.writerows(data)



def main():
	if len(sys.argv) != 2:
		print("Please provide a save dir to load values from")
		exit(1)
	path = sys.argv[1]
	assert os.path.exists(path), "Provided save dir does not exist"
	path = os.path.join(path, "hyper_params.json")
	assert os.path.exists(path), "hyper_params.json does not exist in the provided directory"
	data = load_json(path)
	write_csv(data, sys.argv[1])
	print("\nDone!\n")

if __name__ == '__main__':
	main()

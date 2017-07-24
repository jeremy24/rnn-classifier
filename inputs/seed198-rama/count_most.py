import os
import glob




def main():
	files = glob.glob("./*.ann")
	words = dict()
	print("Have {:,} files:".format(len(files)))
	for filename in files:
		# print("On: ", filename)
		with open(filename, "r") as fin:
			for line in fin:
				line = line.strip().split()
				key = line[1]
				# print(key)
				if key not in words:
					words[key] = 0
				words[key] += 1
	
	for key in sorted(words.keys(), key=lambda x: words[x], reverse=True):
		print("{}  {:,}".format(key, words[key]))
main()

import gc
import re
from enum import Enum
import numpy as np


class LabelTypes(Enum):
	VOWELS = 0
	COMMON_WORDS = 1
	ONE_WORD = 2
	STARTS_WITH_TH = 3
	IMPLICATIONS = 4
	NONE = 5


def labeler(seq, method=None, replace_char=None, words_to_use=10):
	replace_char = chr(1) if replace_char is None else str(replace_char)

	# for testing
	method = LabelTypes.ONE_WORD

	assert len(replace_char) == 1, "Replacement char has a length > 1"
	ret = None

	if not isinstance(method, LabelTypes):
		raise Exception("Method passed to labeler must be a LabelType enum")

	if method == LabelTypes.VOWELS:
		try:
			ret = _vowels(seq, replace_char)
		except Exception as ex:
			raise Exception("Error labeling using vowels:", ex)
	elif method == LabelTypes.COMMON_WORDS:
		try:
			ret = _common_words(seq, replace_char, words_to_use=words_to_use)
		except Exception as ex:
			raise Exception("Error labeling using common words:", ex)
	elif method == LabelTypes.ONE_WORD:
		try:
			ret = _common_words(seq, replace_char, words_to_use=1)
		except Exception as ex:
			raise Exception("Error labeling using one word:", ex)
	elif method == LabelTypes.STARTS_WITH_TH:
		try:
			ret = _starts_with_th(seq, replace_char)
		except Exception as ex:
			raise Exception("Error labeling using starts_with_th words:", ex)
	elif method == LabelTypes.IMPLICATIONS:
		try:
			ret = _implications(seq, replace_char)
		except Exception as ex:
			raise Exception("Error labeling using implications:", ex)
	elif method == LabelTypes.NONE:
		ret = _none(seq, replace_char)

	if ret is None:
		raise Exception("Unable to label data via provided method: ", method)

	assert len(ret) == len(seq)
	return ret


def _none(seq, replace_char):
	return seq

def _common_words(seq, replace_char, words_to_use=5):
	"""Generate labels for a given sequence"""
	print("\nLabeler:")
	print("\tSeq length: {:,} ".format(len(seq)))
	orig_len = len(seq)
	replace_char = str(replace_char)

	# the word list is the top 10 most
	# common words in the sequence
	words = list()
	b = seq.split(" ")
	wc = dict()
	for x in b:
		x = str(x).lower()
		if x not in wc:
			wc[x] = 0
		wc[x] += 1

	print("\nWords being used:")
	for w in sorted(wc, key=wc.get, reverse=True):
		if len(words) == words_to_use:
			break
		# print("\tWord: ", w)
		if len(w) < 3 or not str(w).isalnum():
			# print("\t\tSkipping...")
			continue
		# print("\t")
		w_ = " " + w + " "
		# w_ = w_.replace("\\", "\\\\")
		print("\t[{}]:  {:,}  word len: {}".format(w_, wc[w], len(w_)))
		words.append(w_)

	print("\nGenerating labels based on {} words".format(len(words)))
	# expressions = list()
	ret = np.zeros(len(seq), dtype=np.uint8)

	def make_exp(w, space=True):
		if not space:
			return r"(" + w + ")"
		return r"( " + w + " )"

	expressions = [make_exp(x, space=False) for x in words]

	i = 0
	for word, exp in zip(words, expressions):
		gc.collect()
		replace_string = replace_char * len(word)
		assert len(replace_string) == len(word), "Replace string len doesn't match: [{}]  [{}]".format(
			replace_string, word
		)
		seq = re.sub(exp, replace_string, seq)
		print("\t{:02d}: Done with: [{}]  Used: [{}]  {} {}"
			  .format(i, word, exp, len(word), len(replace_string)))
		assert len(seq) == orig_len, "Lens don't match after word: {}, {:,} != {:,}" \
			.format(exp, len(seq), orig_len)
		i += 1

	print("\n\tDone with all replacements")

	for i in range(len(seq)):
		ret[i] = seq[i] == replace_char

	# assert len(ret) == len(seq), "{} != {}".format(len(ret), orig_len)
	return ret



def bucket_by_size(words):
	sizes = []
	for word in words:
		if len(word) not in sizes:
			sizes.append(len(word))
	return sizes


def _implications(seq, replace_char):
	"""Generate labels for a given sequence"""
	print("\nLabeler:")
	print("\tSeq length: {:,} ".format(len(seq)))
	orig_len = len(seq)
	replace_char = str(replace_char)

	# all words that start with th, whole words only
	find_exp = r"implications\s[A-Za-z0-0)(}{\]\[\-_\s=~]+[\n\.!?]"
	found_words = re.findall(find_exp, seq)
	print("Found words: ", len(found_words))
	print("words[0]")
	print(found_words[0])


	phrases = list()
	for chunk in found_words:
		print(type(chunk))
		if type(chunk) != str:
			for phrase in chunk:
				phrases.append(phrase)
		else:
			phrases.append(chunk)
	print("phrases length: ", len(phrases))

	find_exp = r"important\s[A-Za-z0-0)(}{\]\[\-_\s=~]+[\n\.!?]"
	found_words = re.findall(find_exp, seq)

	for chunk in found_words:
		print(type(chunk))
		if type(chunk) != str:
			for phrase in chunk:
				phrases.append(phrase)
		else:
			phrases.append(chunk)
	print("phrases length: ", len(phrases))


	phrases.sort(key=lambda x: len(x), reverse=True)

	print("\nGenerating labels based on {} sentence chunks".format(len(phrases)))
	sizes = bucket_by_size(phrases)
	print("\tHave {} different sentence lengths".format(len(sizes)))
	print(phrases[0])
	print("\n\n", phrases[1])

	ret = np.zeros(len(seq), dtype=np.uint8)

	total_chars = 0
	for chunk in phrases:
		total_chars += len(chunk)

	# each replace string is [ XXXX ] where X is the replace_char
	i = 0
	print("Finding a labeling {:,} chars from the found phrases".format(total_chars))
	for phrase in phrases:
		gc.collect()
		size = len(phrase)
		replace_string = replace_char * size

		before_len = len(seq)
		seq.replace(phrase, replace_string)

		# seq = re.sub(exp, replace_string, seq, flags=re.IGNORECASE)
		# print("\t{:02d}: Done with phrase\n\t{}".format(i, phrase))
		if i % 20 == 0:
			print("Done with {:04d}".format(i))
		assert len(seq) == before_len, "Lens don't match after word size: {}".format(size)
		i += 1

	print("\n\tDone with all replacements")

	for i in range(len(seq)):
		ret[i] = seq[i] == replace_char

	assert len(ret) == orig_len, "{} != {}".format(len(ret), orig_len)
	return ret



def _starts_with_th(seq, replace_char):
	"""Generate labels for a given sequence"""
	print("\nLabeler:")
	print("\tSeq length: {:,} ".format(len(seq)))
	orig_len = len(seq)
	replace_char = str(replace_char)

	# all words that start with th, whole words only
	find_exp = r"(\b[Tt][Hh][A-Za-z]+\b)"

	words = re.findall(find_exp, seq)
	words = [str(word).lower() for word in words]

	frequency = {}
	for word in words:
		if word not in frequency:
			frequency[word] = 0
		frequency[word] += 1

	# drop all dupes, sort by frequency and then drop all with
	# frequency greater than 20k
	max_frequency = 15000
	min_word_length = 4

	words = list(set(words))
	words.sort(key=lambda x: frequency[x], reverse=True)
	print("Dropping all with frequency less than: ", max_frequency)
	words = [word for word in words if frequency[word] < max_frequency and len(word) >= min_word_length]

	print("\nGenerating labels based on {} words".format(len(words)))
	sizes = bucket_by_size(words)
	print("\tHave {} different word sizes".format(len(sizes)))

	ret = np.zeros(len(seq), dtype=np.uint8)

	# subtract 2 from each length since the "th" adds two chars
	# sort in descending order since we need to replace the smaller words first
	# or the bogger word replacements will pick them up and mess up the
	# number of characters
	sizes.sort()
	assert min_word_length > 2, "Minimum word length must be greater than 2"
	assert np.min(sizes) <= min_word_length, "There is a word that is longer than the minimum word length"
	expressions = ["(\\b[Tt][Hh][A-Za-z]{" + str(min_word_length-2) + ",#}\\b)".replace("#", str(size - 2)) for size in sizes]

	# figure out how many words of each size there are
	size_map = {}
	for key in frequency.keys():
		value = frequency[key]
		if len(key) not in size_map:
			size_map[len(key)] = 0
		size_map[len(key)] += value


	# each replace string is [ XXXX ] where X is the replace_char
	i = 0
	for size, exp in zip(sizes, expressions):
		gc.collect()
		replace_string = replace_char * size
		before_len = len(seq)
		seq = re.sub(exp, replace_string, seq, flags=re.IGNORECASE)
		print("\t{:02d}: Done with word size {:0>2,} ({:8,} occurrences)  Expression:  {}".format(i, size, size_map[size], exp))
		assert len(seq) == before_len, "Lens don't match after word size: {}".format(size)
		i += 1

	print("\n\tDone with all replacements")

	for i in range(len(seq)):
		ret[i] = seq[i] == replace_char

	assert len(ret) == orig_len, "{} != {}".format(len(ret), orig_len)
	return ret


def _vowels(seq, replace_char):
	"""Generate labels for a given sequence"""
	print("\nLabeler:")
	print("\tSeq length: {:,} ".format(len(seq)))
	orig_len = len(seq)
	replace_char = str(replace_char)

	words = ["a", "e", "i", "o", "u"]

	print("\nGenerating labels based on {} words".format(len(words)))

	ret = np.zeros(len(seq), dtype=np.uint8)

	def make_exp(w, space=True):
		if not space:
			return r"(" + w + ")"
		return r"( " + w + " )"

	expressions = [make_exp(x, space=False) for x in words]

	# each replace string is [ XXXX ] where X is the replace_char
	i = 0
	for word, exp in zip(words, expressions):
		gc.collect()
		replace_string = replace_char * len(word)  # (" " + replace_char * len(word) + " ")
		seq = re.sub(exp, replace_string, seq, flags=re.IGNORECASE)
		print("\t{:02d}: Done with: {}".format(i, word))
		i += 1

	print("\n\tDone with all replacements")

	for i in range(len(seq)):
		ret[i] = seq[i] == replace_char

	assert len(ret) == len(seq), "{} != {}".format(len(ret), orig_len)
	return ret

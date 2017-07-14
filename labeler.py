import gc
import re
from enum import Enum
import numpy as np


class LabelTypes(Enum):
	VOWELS = 0
	COMMON_WORDS = 1


def labeler(seq, method=None, replace_char=None, words_to_use=5):
	replace_char = chr(1) if replace_char is None else str(replace_char)

	# for testing
	method = LabelTypes.COMMON_WORDS

	assert len(replace_char) == 1, "Replacement char has a length > 1"
	ret = None

	if not isinstance(method, LabelTypes):
		raise Exception("Method passed to labeler must be a LabelType enum")

	if method == LabelTypes.VOWELS:
		try:
			ret = _vowels(seq, replace_char)
		except Exception as ex:
			raise Exception("Error labeling using vowels:", ex)
	if method == LabelTypes.COMMON_WORDS:
		try:
			ret = _common_words(seq, replace_char, words_to_use=words_to_use)
		except Exception as ex:
			raise Exception("Error labeling using common words:", ex)

	if ret is None:
		raise Exception("Unable to label data via provided method: ", method)

	assert len(ret) == len(seq)
	return ret


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
		if len(w) < 3:
			continue
		w_ = " " + w + " "
		print("\t[{}]:  {:,}".format(w_, wc[w]))
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
		seq = re.sub(exp, replace_string, seq, flags=re.IGNORECASE)
		print("\t{:02d}: Done with: {}".format(i, word))
		i += 1

	print("\n\tDone with all replacements")

	for i in range(len(seq)):
		ret[i] = seq[i] == replace_char

	# assert len(ret) == len(seq), "{} != {}".format(len(ret), orig_len)
	return ret


def _vowels(seq, replace_char):
	"""Generate labels for a given sequence"""
	print("\nLabeler:")
	print("\tSeq length: {:,} ".format(len(seq)))
	orig_len = len(seq)
	replace_char = str(replace_char)
	# words = Prepositions().starts_with("b")
	# words = ["the", "of", "and", "in", "to", "a", "with", "for", "is"]

	# seq = re.sub(r"[\s]{2,}", " ", seq)

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



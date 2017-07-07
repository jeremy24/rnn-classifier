
from decorators import no_dupes

class Prepositions(object):
	preps = list()
	created = False

	def __init__(self, do_multiples=True):
		if not self.created:
			self.fill()
			if do_multiples:
				self.fill_multiples()
			self.drop_dupes()
			self.created = True

	def fill_multiples(self):
		self.preps += ["according to", "adjacent to", "ahead of", "apart from", "as for"]
		self.preps += ["as of", "as per", "as regards", "aside from", "back to", "because of"]
		self.preps += ["close to", "due to", "except for", "far from", "inside of", "instead of"]
		self.preps += ["left of", "near to", "next to", "opposite of", "opposite to"]
		self.preps += ["out from", "out of", "outside of", "owing to", "prior to"]
		self.preps += ["pursuant to", "rather than", "regardless of", "right of"]
		self.preps += ["subsequent to", "such as", "thanks to", "up to"]
		self.preps += ["as far as", "as opposed to", "as soon as", "as well as"]

	def fill(self):
		self.preps += ["aboard", "about", "above", "across", "after", "against"]
		self.preps += ["along", "amid", "among", "anti", "around", "as", "at"]
		self.preps += ["before", "behind", "below", "beneath", "beside", "besides"]
		self.preps += ["between", "beyond", "but", "by", "concerning", "considering"]
		self.preps += ["despite", "down", "during", "except", "excepting", "excluding"]
		self.preps += ["following", "for", "from", "in", "inside", "into", "like"]
		self.preps += ["minus", "near", "of", "off", "on", "onto", "opposite", "outside"]
		self.preps += ["over", "past", "per", "plus", "regarding", "round", "save"]
		self.preps += ["since", "than", "through", "to", "toward", "towards", "under"]
		self.preps += ["underneath", "unlike", "until"]
		self.preps += ["up", "upon", "versus", "via", "with" "within", "without"]

	def drop_dupes(self):
		self.preps = list(set(self.preps))
		self.preps = sorted(self.preps)

	@no_dupes
	@property
	def words(self):
		return self.preps

	@no_dupes
	def starts_with(self, char):
		return [x for x in self.preps if x[0] == char]

	# return map(lambda x: x[0] == char, self.preps)
	@no_dupes
	def len_between(self, lower=None, upper=None):
		"""Gets words where  lower < len(word) < upper"""
		lower = 0 if lower is None else int(lower)
		upper = float("inf") if upper is None else int(upper)
		ret = [x for x in self.preps if lower <= len(x) <= upper]
		return ret

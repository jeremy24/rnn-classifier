from __future__ import absolute_import
from __future__ import print_function

import functools

import tensorflow as tf

"""The double wrap decorators are found at https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2"""


def doublewrap(function):
	"""
	A decorator decorator, allowing to use the decorator to be used without
	parentheses if not arguments are provided. All arguments must be optional.
	"""

	@functools.wraps(function)
	def decorator(*args, **kwargs):
		if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
			return function(args[0])
		else:
			return lambda wrapee: function(wrapee, *args, **kwargs)

	return decorator


@doublewrap
def ifnotdefined(function):
	"""
	:param function:
	A decorator for functions that defines a TensorFlow OP. The wrapped function will
	 only be called once to avoid multiple things being added to the graph. This will not
	 define a TensorFlow variable_scope, if you want a variable scope use @define_scope
	"""
	attribute = "_cache_" + function.__name__

	@property
	@functools.wraps(function)
	def decorator(self):
		if not hasattr(self, attribute):
			setattr(self, attribute, function(self))
		return getattr(self, attribute)

	return decorator


@doublewrap
def define_scope_no_property(function, scope=None, *args, **kwargs):
	"""
	A decorator for functions that define TensorFlow operations. The wrapped
	function will only be executed once. Subsequent calls to it will directly
	return the result so that operations are added to the graph only once.
	This does not create a tf.name_scope variable scope
	"""
	attribute = '_cache_' + function.__name__
	name = scope or function.__name__

	@functools.wraps(function)
	def decorator(self):
		if not hasattr(self, attribute):
			with tf.variable_scope(name, *args, **kwargs):
				setattr(self, attribute, function(self))
		return getattr(self, attribute)

	return decorator


@doublewrap
def as_int32(function, name="to_in32", *args, **kwargs):
	"""
	:param function (Passed Automatically)
	:param name [optional]
	:return: The results of the function cast to tf.int32
	"""

	@functools.wraps(function)
	def decorator(self):
		return tf.to_int32(function, name=name)

	return decorator


@doublewrap
def as_float(function, name="to_float", *args, **kwargs):
	"""
	:param function (Passed Automatically)
	:param name [optional]
	:return: The results of the function cast to tf.float via tf.to_float
	"""

	@functools.wraps(function)
	def decorator(self):
		return tf.to_float(function(self), name=name)

	return decorator


@doublewrap
def format_float(function, precision=10, *args, **kwargs):
	"""
	:param function (Passed Automatically)
	:param precision
	:return: The results of the function rounded to precision
	"""

	@functools.wraps(function)
	def decorator(self):
		return round(function(self), precision)

	return decorator


@doublewrap
def no_dupes(function, *args, **kwargs):
	"""
	:return: The results of the function as a list without any dupes
	"""

	@functools.wraps(function)
	def decorator(self):
		return list(set(list(function(self))))

	return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
	"""
	A decorator for functions that define TensorFlow operations. The wrapped
	function will only be executed once. Subsequent calls to it will directly
	return the result so that operations are added to the graph only once.
	The operations added by the function live within a tf.variable_scope(). If
	this decorator is used with arguments, they will be forwarded to the
	variable scope. The scope name defaults to the name of the wrapped
	function.
	"""
	attribute = '_cache_' + function.__name__
	name = scope or function.__name__

	@property
	@functools.wraps(function)
	def decorator(self):
		if not hasattr(self, attribute):
			with tf.variable_scope(name, *args, **kwargs):
				setattr(self, attribute, function(self))
		return getattr(self, attribute)

	return decorator

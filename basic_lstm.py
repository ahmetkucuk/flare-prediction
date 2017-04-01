import tensorflow as tf
from tensorflow.contrib import rnn


class BasicLSTMModel(object):

	def __init__(self, n_input=14, n_steps=60, n_hidden=128, n_classes=2):
		# tf Graph input
		self.x = tf.placeholder("float", [None, n_steps, n_input])
		self.y = tf.placeholder("float", [None, n_classes])

		# Define weights
		self.weights = {
			'out': tf.Variable(tf.random_normal([n_hidden, n_classes])),
			'ensemble': tf.Variable(tf.random_normal([2]))
		}

		self.biases = {
			'out': tf.Variable(tf.random_normal([n_classes])),
			'ensemble': tf.Variable(tf.random_normal([2]))
		}

		# Prepare data shape to match `rnn` function requirements
		# Current data input shape: (batch_size, n_steps, n_input)
		# Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

		# Permuting batch_size and n_steps
		x = tf.transpose(self.x, [1, 0, 2])
		# Reshaping to (n_steps*batch_size, n_input)
		x = tf.reshape(x, [-1, n_input])
		# Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
		x = tf.split(x, n_steps, 0)

		# Define a lstm cell with tensorflow
		lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

		# Get lstm cell output
		outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

		# Linear activation, using rnn inner loop last output
		self.preds = tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']

	def get_preds(self):
		return self.preds

	def get_placeholders(self):
		return self.x, self.y
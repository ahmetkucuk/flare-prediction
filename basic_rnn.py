import tensorflow as tf
from tensorflow.contrib import rnn


class BasicRNNModel(object):

	def __init__(self, n_input=14, n_steps=60, n_hidden=256, n_classes=2, n_cells=2, dropout_keep_prob=0.5, is_lstm=True):
		# tf Graph input
		self.x = tf.placeholder(tf.float32, [None, n_steps, n_input])
		self.y = tf.placeholder(tf.float32, [None, n_classes])
		self.dropout = tf.placeholder(tf.float32)

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
		#lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

		if is_lstm:
			stacked_cells = rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.BasicLSTMCell(n_hidden, forget_bias=1.0), dropout_keep_prob) for _ in range(n_cells)])
		else:
			stacked_cells = rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.GRUCell(n_hidden), dropout_keep_prob) for _ in range(n_cells)])
		# Get lstm cell output
		outputs, states = rnn.static_rnn(stacked_cells, x, dtype=tf.float32)

		# Linear activation, using rnn inner loop last output
		self.preds = tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']

	def get_preds(self):
		return self.preds

	def get_placeholders(self):
		return self.x, self.y, self.dropout

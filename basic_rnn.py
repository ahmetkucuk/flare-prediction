import tensorflow as tf
import tensorflow.contrib.rnn as rnn
#from tensorflow.python.ops import rnn


class BasicRNNModel(object):

	def __init__(self, n_input=14, n_steps=60, n_hidden=256, n_classes=2, n_cells=2, cell_type="LSTM"):
		# tf Graph input
		self.x = tf.placeholder(tf.float32, [None, n_steps, n_input])
		self.dropout = tf.placeholder(tf.float32)
		self.n_hidden = n_hidden

		# Define weights
		self.weights = {
			'out': tf.Variable(tf.random_normal([n_hidden, n_classes])),
		}

		self.biases = {
			'out': tf.Variable(tf.random_normal([n_classes])),
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

		if cell_type == "LSTM":
			stacked_cells = rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.BasicLSTMCell(n_hidden, forget_bias=1.0), self.dropout) for _ in range(n_cells)])
		elif cell_type == "GRU":
			stacked_cells = rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.GRUCell(n_hidden), self.dropout) for _ in range(n_cells)])
		elif cell_type == "LSTM_BLOCK":
			stacked_cells = rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.LSTMBlockCell(n_hidden), self.dropout) for _ in range(n_cells)])
		elif cell_type == "LAYER_NORM_LSTM":
			stacked_cells = rnn.MultiRNNCell([rnn.LayerNormBasicLSTMCell(n_hidden, dropout_keep_prob=self.dropout) for _ in range(n_cells)])
		elif cell_type == "BASIC_RNN":
			stacked_cells = rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.BasicRNNCell(n_hidden), self.dropout) for _ in range(n_cells)])
		elif cell_type == "NAS":
			stacked_cells = rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.NASCell(n_hidden), self.dropout) for _ in range(n_cells)])
		else:
			print("cell type not recognized: " + cell_type)
			exit()
		# Get lstm cell output
		outputs, states = rnn.static_rnn(stacked_cells, x, dtype=tf.float32)
		#
		# output_flattened = tf.reshape(outputs, [-1, n_hidden])
		# output_logits = tf.add(tf.matmul(output_flattened, self.weights['last']), self.weights['out'])
		# output_all = tf.nn.sigmoid(output_logits)
		# output_reshaped = tf.reshape(output_all,[-1,n_steps,n_classes])
		# output_last = tf.gather(tf.transpose(output_reshaped,[1,0,2]), n_steps - 1)
		self.output = outputs[-1]
		self.preds = tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']

	def get_preds(self):
		return self.preds

	def get_output(self):
		return self.output

	def get_placeholders(self):
		return self.x, self.dropout

	def get_n_hidden(self):
		return self.n_hidden
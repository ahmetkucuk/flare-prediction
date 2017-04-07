import sys
import tensorflow as tf
from flare_dataset import get_prior12_span12
from basic_rnn import BasicRNNModel
from train_basic_lstm import TrainRNN
from configurations import get_norm_func


FLAGS = tf.app.flags.FLAGS

n_input = 14
n_steps = 60
n_hidden = 256
n_classes = 2

learning_rate = 0.0001
training_iters = 300000
batch_size = 20
display_step = 100

n_cells = 1

is_lstm = False

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
						   """Directory where to write event logs """
						   """and checkpoint.""")

tf.app.flags.DEFINE_string('dataset_dir', '/home/ahmet/Documents/Research/Time_Series/ARDataLarge',
						   """Directory where to write event logs """
						   """and checkpoint.""")

tf.app.flags.DEFINE_string('location', 'local',
						   """Where does the experiment started""")

tf.app.flags.DEFINE_string('experiment_name', 'not_specified',
						   """Name of the experiment to be shown in tensorboard""")

tf.app.flags.DEFINE_string('norm_type', 'min_max',
						   """Normalization Function""")

tf.app.flags.DEFINE_integer('n_input', 14,
							"""Number of inputs - number of feature.""")

tf.app.flags.DEFINE_integer('n_steps', 60,
							"""Number of inputs - number of feature.""")

tf.app.flags.DEFINE_boolean('n_hidden', 128,
							"""Number of hidden units in each RNN cell.""")

tf.app.flags.DEFINE_boolean('n_classes', 2,
							"""Number of class to be classified.""")

tf.app.flags.DEFINE_boolean('learning_rate', 0.0001,
							"""Learning rate.""")

tf.app.flags.DEFINE_boolean('batch_size', 20,
							"""Batch Size.""")

tf.app.flags.DEFINE_boolean('training_iters', 300000,
							"""Number of iterations.""")

tf.app.flags.DEFINE_boolean('n_cells', 1,
							"""Number RNN cells.""")

tf.app.flags.DEFINE_boolean('is_lstm', False,
							"""Pick if lstm or gru cell.""")


def main(argv=None):


	norm_func = get_norm_func(FLAGS.norm_type)

	dataset = get_prior12_span12(data_root=FLAGS.dataset_dir, norm_func=norm_func)

	lstm = BasicRNNModel(n_input=FLAGS.n_input, n_steps=FLAGS.n_steps, n_hidden=FLAGS.n_hidden, n_classes=FLAGS.n_classes, n_cells=FLAGS.n_cells, is_lstm=FLAGS.is_lstm)
	train_lstm = TrainRNN(lstm, dataset, model_dir=FLAGS.train_dir, learning_rate=FLAGS.learning_rate, training_iters=FLAGS.training_iters, batch_size=FLAGS.batch_size, display_step=FLAGS.display_step)
	train_lstm.train()

if __name__ == "__main__":
	tf.app.run()

	#run(["local", "min_max", "norm_min_max"])
	#run(["local", "z_score", "norm_z_score"])
	#run(["local", "zero_center", "norm_zero_center"])
	#run(["local", "pca_whiten", "norm_pca_whiten"])

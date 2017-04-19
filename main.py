import sys
import tensorflow as tf
from flare_dataset import get_prior12_span12
from flare_dataset import get_data
from basic_rnn import BasicRNNModel
from train_basic_lstm import TrainRNN
from configurations import get_norm_func
from configurations import get_feature_indexes


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
						   """Directory where to write event logs """
						   """and checkpoint.""")

tf.app.flags.DEFINE_string('dataset_dir', '/home/ahmet/Documents/Research/Time_Series/ARDataLarge',
						   """Directory where to write event logs """
						   """and checkpoint.""")

tf.app.flags.DEFINE_string('experiment_name', 'not_specified',
						   """Name of the experiment to be shown in tensorboard""")

tf.app.flags.DEFINE_string('norm_type', 'min_max',
						   """Normalization Function""")

tf.app.flags.DEFINE_integer('n_input', 14,
							"""Number of inputs - number of feature.""")

tf.app.flags.DEFINE_integer('n_steps', 60,
							"""Number of inputs - number of feature.""")

tf.app.flags.DEFINE_integer('n_hidden', 128,
							"""Number of hidden units in each RNN cell.""")

tf.app.flags.DEFINE_integer('n_classes', 2,
							"""Number of class to be classified.""")

tf.app.flags.DEFINE_float('learning_rate', 0.0001,
							"""Learning rate.""")

tf.app.flags.DEFINE_float('dropout', 0.7,
						  """Learning rate.""")

tf.app.flags.DEFINE_integer('batch_size', 20,
							"""Batch Size.""")

tf.app.flags.DEFINE_integer('training_iters', 20000,
							"""Number of iterations.""")

tf.app.flags.DEFINE_integer('n_cells', 1,
							"""Number RNN cells.""")

tf.app.flags.DEFINE_string('cell_type', "BASIC_RNN",
							"""Pick if lstm or gru cell.""")

tf.app.flags.DEFINE_integer('augmentation_type', 0,
							"""Pick if augment type on training data.""")

tf.app.flags.DEFINE_string('dataset_name', "12_24",
							"""Pick if should augment training data.""")

tf.app.flags.DEFINE_integer('display_step', 1000,
							"""Print results after x steps.""")

tf.app.flags.DEFINE_string('feature_indexes', "1",
							"""Feature that will be included.""")


def main(argv=None):

	norm_func = get_norm_func(FLAGS.norm_type)
	feature_indexes = get_feature_indexes(FLAGS.feature_indexes)

	dataset = get_data(name=FLAGS.dataset_name, data_root=FLAGS.dataset_dir, norm_func=norm_func, augmentation_type=FLAGS.augmentation_type, feature_indexes=feature_indexes)
	print("Length of Dataset: " + str(dataset.size()))

	lstm = BasicRNNModel(n_input=FLAGS.n_input, n_steps=FLAGS.n_steps, n_hidden=FLAGS.n_hidden,
						n_classes=FLAGS.n_classes, n_cells=FLAGS.n_cells, cell_type=FLAGS.cell_type)

	train_lstm = TrainRNN(lstm, dataset, model_dir=FLAGS.train_dir, learning_rate=FLAGS.learning_rate,
						training_iters=FLAGS.training_iters, batch_size=FLAGS.batch_size,
						display_step=FLAGS.display_step, dropout_val=FLAGS.dropout)
	train_lstm.train()

if __name__ == "__main__":
	tf.app.run()

	#run(["local", "min_max", "norm_min_max"])
	#run(["local", "z_score", "norm_z_score"])
	#run(["local", "zero_center", "norm_zero_center"])
	#run(["local", "pca_whiten", "norm_pca_whiten"])

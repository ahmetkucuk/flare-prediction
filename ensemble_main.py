
import tensorflow as tf
from basic_rnn import BasicRNNModel



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

tf.app.flags.DEFINE_boolean('is_lstm', False,
                            """Pick if lstm or gru cell.""")

tf.app.flags.DEFINE_boolean('should_augment', True,
                            """Pick if should augment training data.""")

tf.app.flags.DEFINE_string('dataset_name', "12_24",
                           """Pick if should augment training data.""")

tf.app.flags.DEFINE_integer('display_step', 1000,
                            """Print results after x steps.""")

def main(args=None):

    with tf.variable_scope("ensemble"):

        with tf.variable_scope("model1"):
            lstm1 = BasicRNNModel(n_input=FLAGS.n_input, n_steps=FLAGS.n_steps, n_hidden=FLAGS.n_hidden,
                             n_classes=FLAGS.n_classes, n_cells=FLAGS.n_cells, is_lstm=FLAGS.is_lstm)

        with tf.variable_scope("model2"):
            lstm2 = BasicRNNModel(n_input=FLAGS.n_input, n_steps=FLAGS.n_steps, n_hidden=FLAGS.n_hidden,
                                 n_classes=FLAGS.n_classes, n_cells=FLAGS.n_cells, is_lstm=FLAGS.is_lstm)




if __name__ == "__main__":
    tf.app.run()
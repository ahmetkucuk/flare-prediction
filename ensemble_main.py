
import tensorflow as tf
from basic_rnn import BasicRNNModel
from flare_dataset import get_multi_data
from flare_dataset import get_multi_feature
from configurations import get_norm_func
from ensemble_rnn import EnsembleRNN

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/Users/ahmetkucuk/Documents/Research/Flare_Prediction/Tensorboard/FlarePrediction/ensemble',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_string('dataset_dir', '/Users/ahmetkucuk/Documents/Research/Flare_Prediction/ARDataLarge',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_string('experiment_name', 'not_specified',
                           """Name of the experiment to be shown in tensorboard""")

tf.app.flags.DEFINE_string('norm_type', 'z_score',
                           """Normalization Function""")

tf.app.flags.DEFINE_integer('n_input', 14,
                            """Number of inputs - number of feature.""")

tf.app.flags.DEFINE_integer('n_steps', 120,
                            """Number of inputs - number of feature.""")

tf.app.flags.DEFINE_integer('n_hidden', 256,
                            """Number of hidden units in each RNN cell.""")

tf.app.flags.DEFINE_integer('n_classes', 2,
                            """Number of class to be classified.""")

tf.app.flags.DEFINE_float('learning_rate', 0.0001,
                          """Learning rate.""")

tf.app.flags.DEFINE_float('dropout', 0.7,
                          """Learning rate.""")

tf.app.flags.DEFINE_integer('batch_size', 20,
                            """Batch Size.""")

tf.app.flags.DEFINE_integer('training_iters', 40000,
                            """Number of iterations.""")

tf.app.flags.DEFINE_boolean('is_lstm', False,
                            """Pick if lstm or gru cell.""")

tf.app.flags.DEFINE_string('dataset_name', "24_24",
                           """Pick if should augment training data.""")

tf.app.flags.DEFINE_integer('display_step', 100,
                            """Print results after x steps.""")

tf.app.flags.DEFINE_boolean('is_multi_feature', True,
                            """Print results after x steps.""")

tf.app.flags.DEFINE_integer('f1', 13,
                           """First Feature.""")

tf.app.flags.DEFINE_integer('f2', 12,
                            """Second Feature.""")

tf.app.flags.DEFINE_string('m1_cell', "GRU",
                            """First Cell Type.""")

tf.app.flags.DEFINE_string('m2_cell', "GRU",
                            """Second Cell Type.""")

tf.app.flags.DEFINE_string('augmentation_types', "-1",
                            """Augmentation Types.""")


def main(args=None):
    norm_func = get_norm_func(FLAGS.norm_type)

    augmentation_types = [int(i) for i in FLAGS.augmentation_types.split(",")]

    if FLAGS.is_multi_feature:
        print("Ensemble Started with Multi Feature")
        dataset = get_multi_feature(data_root=FLAGS.dataset_dir, norm_func=norm_func, augmentation_types=augmentation_types, f1=[FLAGS.f1], f2=[FLAGS.f2], dataname=FLAGS.dataset_name)
    else:
        print("Ensemble Started with Multi Dataset")
        dataset = get_multi_data(data_root=FLAGS.dataset_dir, norm_func=norm_func, augmentation_types=augmentation_types, feature_indexes=[13])

    train_lstm = EnsembleRNN(dataset, model_dir=FLAGS.train_dir, learning_rate=FLAGS.learning_rate,
                          training_iters=FLAGS.training_iters, batch_size=FLAGS.batch_size,
                          display_step=FLAGS.display_step, dropout_val=FLAGS.dropout, n_hidden=FLAGS.n_hidden,
                             n_steps=FLAGS.n_steps, m1_cell=FLAGS.m1_cell, m2_cell=FLAGS.m2_cell)
    train_lstm.train()


if __name__ == "__main__":
    tf.app.run()
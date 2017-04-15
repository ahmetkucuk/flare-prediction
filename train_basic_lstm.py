from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np


class TrainRNN(object):

	def __init__(self, model, dataset, model_dir, learning_rate=0.001, training_iters=300000, batch_size=9, display_step=100, dropout_val=0.5):

		self.dataset = dataset
		self.model_dir = model_dir
		self.learning_rate = learning_rate
		self.training_iters = training_iters
		self.batch_size = batch_size
		self.display_step = display_step
		self.dropout_val = dropout_val

		self.x, self.dropout = model.get_placeholders()
		self.y = tf.placeholder(tf.float32, [None, 2])
		self.preds = model.get_preds()
		self.output = model.get_output()

		with tf.name_scope("evaluations"):
			# Define loss and optimizer

			with tf.name_scope("loss"):
				self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.preds, labels=self.y))
				self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
				tf.summary.scalar('loss', self.cost)

			with tf.name_scope("prediction"):
				correct_pred = tf.equal(tf.argmax(self.preds, 1), tf.argmax(self.y, 1))
				self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
				tf.summary.scalar('accuracy', self.accuracy)

	def create_embeddings(self, sess, summary_writer):
		EMB = np.zeros([100, 64], dtype='float32')

		metadata_file = open(self.model_dir + '/train/metadata.tsv', 'w')
		for i in range(100):
			batch_x, batch_y = self.dataset.next_batch(1)
			o = sess.run(self.output, feed_dict={self.x: batch_x, self.y: batch_y, self.dropout: 1})
			EMB[i] = o[0]
			if batch_y[0][0] == 0:
				metadata_file.write("Flare\n")
			else:
				metadata_file.write("NONFlare\n")
		metadata_file.close()

		embedding_var = tf.Variable(EMB,  name='Embedding_of_output')
		sess.run(embedding_var.initializer)
		config = projector.ProjectorConfig()
		embedding = config.embeddings.add()
		embedding.tensor_name = embedding_var.name
		embedding.metadata_path = self.model_dir + '/train/metadata.tsv'

		projector.visualize_embeddings(summary_writer, config)
		saver = tf.train.Saver([embedding_var])
		saver.save(sess, self.model_dir + '/train/model.ckpt', 1)

	def train(self):
		init = tf.global_variables_initializer()
		sess = tf.Session()
		#with tf.Session() as sess:

		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(self.model_dir + '/train', sess.graph)
		test_writer = tf.summary.FileWriter(self.model_dir + '/test', sess.graph)
		sess.run(init)
		step = 1
		# Keep training until reach max iterations
		while step < self.training_iters:
			batch_x, batch_y = self.dataset.next_batch(self.batch_size)

			sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y, self.dropout: self.dropout_val})
			if step % self.display_step == 0:

				summary, acc = sess.run([merged, self.accuracy], feed_dict={self.x: batch_x, self.y: batch_y, self.dropout: 1})
				train_writer.add_summary(summary=summary, global_step=step)

				summary, loss = sess.run([merged, self.cost], feed_dict={self.x: batch_x, self.y: batch_y, self.dropout: 1})
				train_writer.add_summary(summary=summary, global_step=step)

				print("Iter " + str(step) + ", Minibatch Loss= " + \
					  "{:.6f}".format(loss) + ", Training Accuracy= " + \
					  "{:.5f}".format(acc))

				test_data, test_label = self.dataset.get_test()
				summary, accuracy = sess.run([merged, self.accuracy], feed_dict={self.x: test_data, self.y: test_label, self.dropout: 1})
				test_writer.add_summary(summary=summary, global_step=step)
				print("Test Accuracy: {:.6f}".format(accuracy))
				train_writer.flush()
				test_writer.flush()
			step += 1
		train_writer.close()
		test_writer.close()
		self.create_embeddings(sess, train_writer)
		sess.close()

		print("Optimization Finished!")



from __future__ import print_function

import tensorflow as tf

class TrainRNN(object):

	def __init__(self, model, dataset, model_dir, learning_rate=0.001, training_iters=300000, batch_size=9, display_step=100):

		self.dataset = dataset
		self.model_dir = model_dir
		self.learning_rate = learning_rate
		self.training_iters = training_iters
		self.batch_size = batch_size
		self.display_step = display_step

		self.x, self.y, self.dropout = model.get_placeholders()
		self.preds = model.get_preds()

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
		while step * self.batch_size < self.training_iters:
			batch_x, batch_y = self.dataset.next_batch(self.batch_size)

			sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y, self.dropout: 0.5})
			if step % self.display_step == 0:

				summary, acc = sess.run([merged, self.accuracy], feed_dict={self.x: batch_x, self.y: batch_y, self.dropout: 1})
				train_writer.add_summary(summary=summary, global_step=step)

				summary, loss = sess.run([merged, self.cost], feed_dict={self.x: batch_x, self.y: batch_y, self.dropout: 1})
				train_writer.add_summary(summary=summary, global_step=step)

				print("Iter " + str(step*self.batch_size) + ", Minibatch Loss= " + \
					  "{:.6f}".format(loss) + ", Training Accuracy= " + \
					  "{:.5f}".format(acc))

				test_data, test_label = self.dataset.get_test()
				summary, accuracy = sess.run([merged, self.accuracy], feed_dict={self.x: test_data, self.y: test_label, self.dropout: 1})
				test_writer.add_summary(summary=summary, global_step=step)
				print("Test Accuracy: {:.6f}".format(accuracy))
			step += 1
		print("Optimization Finished!")


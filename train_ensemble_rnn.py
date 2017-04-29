
import tensorflow as tf
from basic_rnn import BasicRNNModel


class EnsembleRNN(object):

	def __init__(self, dataset, model_dir, learning_rate=0.001, training_iters=30000, batch_size=20, display_step=100, dropout_val=0.7, n_hidden=128, n_steps=60,  m1_cell="GRU", m2_cell="GRU"):

		with tf.variable_scope("ensemble"):

			self.dataset = dataset
			self.model_dir = model_dir
			self.learning_rate = learning_rate
			self.training_iters = training_iters
			self.batch_size = batch_size
			self.display_step = display_step
			self.dropout_val = dropout_val


			self.weights = {
				'ensemble_weights': tf.Variable(tf.random_normal([n_hidden*2, 2])),
			}

			self.biases = {
				'ensemble_biases': tf.Variable(tf.random_normal([2])),
			}

			with tf.variable_scope("model1"):
				lstm1 = BasicRNNModel(n_input=1, n_steps=n_steps, n_hidden=n_hidden,
									  n_classes=2, n_cells=1, cell_type=m1_cell)

			with tf.variable_scope("model2"):
				lstm2 = BasicRNNModel(n_input=1, n_steps=n_steps, n_hidden=n_hidden,
									  n_classes=2, n_cells=1, cell_type=m2_cell)

			self.x1, self.dropout1 = lstm1.get_placeholders()
			self.output1 = lstm1.get_output()

			self.x2, self.dropout2 = lstm2.get_placeholders()
			self.output2 = lstm2.get_output()
			self.y = tf.placeholder(tf.float32, [None, 2])

			with tf.name_scope("evaluations"):
				self.output = tf.concat([self.output1, self.output2], axis=1, name="model_preds_concat")
				self.preds = tf.matmul(self.output, self.weights['ensemble_weights']) + self.biases['ensemble_biases']

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
		output_file = open(self.model_dir + '/test/validation_accuracy.txt', "w")
		epoch = 0
		while step < self.training_iters:
			batch_x1, batch_x2, batch_y = self.dataset.next_batch(self.batch_size)

			sess.run(self.optimizer, feed_dict={self.x1: batch_x1, self.x2: batch_x2, self.y: batch_y, self.dropout1: self.dropout_val, self.dropout2: self.dropout_val})
			if step % self.display_step == 0:

				summary, acc = sess.run([merged, self.accuracy], feed_dict={self.x1: batch_x1, self.x2: batch_x2, self.y: batch_y, self.dropout1: 1, self.dropout2: 1})
				train_writer.add_summary(summary=summary, global_step=step)

				summary, loss = sess.run([merged, self.cost], feed_dict={self.x1: batch_x1, self.x2: batch_x2, self.y: batch_y, self.dropout1: 1, self.dropout2: 1})
				train_writer.add_summary(summary=summary, global_step=step)

				print("Iter " + str(step) + ", Minibatch Loss= " + \
					  "{:.6f}".format(loss) + ", Training Accuracy= " + \
					  "{:.5f}".format(acc))

				test_data1, test_data2, test_label = self.dataset.get_multi_test()
				summary, accuracy = sess.run([merged, self.accuracy], feed_dict={self.x1: test_data1, self.x2: test_data2, self.y: test_label, self.dropout1: 1, self.dropout2: 1})
				test_writer.add_summary(summary=summary, global_step=step)
				print("Test Accuracy: {:.6f}".format(accuracy))
				if (step*self.batch_size) / self.dataset.size() > epoch:
					epoch = epoch + 1
					output_file.write(str(step) + "\t" + str(accuracy) + "\n")
				train_writer.flush()
				test_writer.flush()
			step += 1
		output_file.close()
		train_writer.close()
		test_writer.close()
		print("Optimization Finished!")
import os, time
import numpy as np
import tensorflow as tf
from network_model import Network
from dataset import Dataset

class TrainModel:
	def train(self):
		BATCH_SIZE = 256

		network = Network()

		dataset = Dataset(folder='data{}_{}'.format(network.IMAGE_HEIGHT, network.IMAGE_WIDTH), batch_size=BATCH_SIZE)

		inputs, targets = dataset.next_batch()

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			network.loadRecog(sess)

			saver = tf.train.Saver(var_list=network.variables)

			#Pre treino
			n_epochs = 50
			print("Starting Pretrain")
			for epoch_i in range(n_epochs):
				dataset.reset_batch_pointer()

				for batch_i in range(dataset.num_batches_in_epoch()):
					batch_num = epoch_i * dataset.num_batches_in_epoch() + batch_i + 1

					start = time.time()
					batch_inputs, batch_targets = dataset.next_batch(pretrain=True)
					batch_inputs = np.reshape(batch_inputs,(dataset.batch_size, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))
					batch_targets = np.reshape(batch_targets,(dataset.batch_size, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))

					batch_inputs = np.multiply(batch_inputs, 1.0 / 255)
					batch_targets = np.multiply(batch_targets, 1.0 / 255)

					cost1, rec,_ = sess.run([network.cost_mse, network.cost_rec, network.train_op_mse],feed_dict={network.inputs: batch_inputs, network.targets: batch_targets, network.is_training: True})
					end = time.time()
					print('{}/{}, epoch: {}, mse/rec: {}/{}, batch time: {}'.format(batch_num, n_epochs * dataset.num_batches_in_epoch(), epoch_i, cost1, rec, round(end - start,5)))        

			test_accuracies = []
			test_mse = []
			n_epochs = 3000
			global_start = time.time()
			print("Starting Train")
			for epoch_i in range(n_epochs):
				dataset.reset_batch_pointer()

				for batch_i in range(dataset.num_batches_in_epoch()):
					batch_num = epoch_i * dataset.num_batches_in_epoch() + batch_i + 1

					start = time.time()
					batch_inputs, batch_targets = dataset.next_batch()
					batch_inputs = np.reshape(batch_inputs,(dataset.batch_size, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))
					batch_targets = np.reshape(batch_targets,(dataset.batch_size, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))

					batch_inputs = np.multiply(batch_inputs, 1.0 / 255)
					batch_targets = np.multiply(batch_targets, 1.0 / 255)

					if batch_num % 5 == 0:
						cost1, rec, _ = sess.run([network.cost_mse, network.cost_rec, network.train_op_rec],feed_dict={network.inputs: batch_inputs, network.targets: batch_targets, network.is_training: True})
					else:
						cost1, rec, _ = sess.run([network.cost_mse, network.cost_rec, network.train_op_mse],feed_dict={network.inputs: batch_inputs, network.targets: batch_targets, network.is_training: True})

					end = time.time()
					print('{}/{}, epoch: {}, mse/recog: {}/{}, batch time: {}'.format(batch_num, n_epochs * dataset.num_batches_in_epoch(), epoch_i, cost1, rec, round(end - start,5)))

					if batch_num % 2000 == 0 or batch_num == n_epochs * dataset.num_batches_in_epoch():
						test_inputs, test_targets = dataset.valid_set
						test_inputs, test_targets = test_inputs[:100], test_targets[:100]

						test_inputs = np.reshape(test_inputs, (-1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))
						test_targets = np.reshape(test_targets, (-1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))
						test_inputs = np.multiply(test_inputs, 1.0 / 255)
						test_targets = np.multiply(test_targets, 1.0 / 255)

						test_accuracy, mse = sess.run([network.accuracy, network.mse],
														  feed_dict={network.inputs: test_inputs,network.targets: test_targets,network.is_training: False})

						#summary_writer.add_summary(summary, batch_num)

						print('Step {}, test accuracy: {}'.format(batch_num, test_accuracy))
						min_mse = (0,0)
						test_accuracies.append((test_accuracy, batch_num))
						test_mse.append((mse, batch_num))
						print("Accuracies in time: ", [test_accuracies[x][0] for x in range(len(test_accuracies))])
						print("MSE in time: ", [test_mse[x][0] for x in range(len(test_mse))])
						max_acc = max(test_accuracies)
						min_mse = min(test_mse)
						print("Best accuracy: {} in batch {}".format(max_acc[0], max_acc[1]))
						print("Total time: {}".format(time.time() - global_start))

						if (batch_num > 50000):
							print("saving model...")
							saver.save(sess,os.path.join("save","checkpoint.data"))

if __name__ == '__main__':
	t = TrainModel()
	t.train()
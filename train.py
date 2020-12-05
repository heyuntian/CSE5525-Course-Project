from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as kr
import time

from utils import readEmbeddings, readTextEmbeddings, readInfo

def args_parser():
	parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
						conflict_handler='resolve')
	parser.add_argument('--dir', required=False, default='./processed_data/',
						help='directory of preprocessed data')
	parser.add_argument('--graphemb', required=False, default='embeddings.txt.txt',
						help='filename of graph embeddings')
	parser.add_argument('--textemb', required=False, default='doc2vec_embedding.txt',
						help='filename of graph embeddings')
	parser.add_argument('--setting', required=False, type=int, default=0,
						help='input for classifier. 0 for graph embeddings only, 1 for text embedding only, 2 for both.')
	parser.add_argument('--epoch', required=False, type=int, default=5,
						help='number of epochs for training')
	parser.add_argument('--batch-size', required=False, type=int, default=1000,
						help='batch_size for training')
	parser.add_argument('--lr', required=False, type=float, default=1e-3,
						help='learning rate')
	parser.add_argument('--easy', action='store_true',
						help='use binary rating instead of decimal')
	parser.add_argument('--finalemb', required=False, type=int, default=128,
						help='final dimensionality of movie embeddings')
	args = parser.parse_args()
	return args

def getIndInfo(args):
	num_movies, num_genres, num_cast, num_users, total = readInfo(args)
	movie_base = 0
	user_base = num_movies + num_genres + num_cast
	return num_users, user_base, num_movies, movie_base

class DataLoader():
	def __init__(self, args, user_emb, movie_emb, user_base, movie_base):
		self.movie_emb = movie_emb
		self.user_emb = user_emb
		self.easy = args.easy

		filename = "/".join([args.dir, "rating_train.csv"])
		train_df = pd.read_csv(filename)
		self.train_uid = np.array(train_df["uId"], dtype=np.int32) - user_base
		self.train_mid = np.array(train_df["mId"], dtype=np.int32) - movie_base
		self.train_binary = np.array(train_df["binary"], dtype=np.int32)
		self.train_rating = np.array(train_df["rating"] * 2 - 1, dtype=np.int32)

		filename = "/".join([args.dir, "rating_test.csv"])
		test_df = pd.read_csv(filename)
		self.test_uid = np.array(test_df["uId"], dtype=np.int32) - user_base
		self.test_mid = np.array(test_df["mId"], dtype=np.int32) - movie_base
		self.test_binary = np.array(test_df["binary"], dtype=np.int32)
		self.test_rating = np.array(test_df["rating"] * 2 - 1, dtype=np.int32)

		self.num_train_data, self.num_test_data = self.train_uid.shape[0], self.test_uid.shape[0]

		print("**********\nDataLoader")
		print("num_train_data: %d, num_test_data: %d"%(self.num_train_data, self.num_test_data))
		print("data_dim: %d"%(self.user_emb.shape[1] + self.movie_emb.shape[1]))

	def get_batch(self, batch_size):
		index = np.random.randint(0, self.num_train_data, batch_size)
		uIds = self.train_uid[index]
		mIds = self.train_mid[index]
		train_data = np.concatenate([self.user_emb[uIds], self.movie_emb[mIds]], axis=1)
		train_label = self.train_binary[index] if self.easy else self.train_rating[index]
		return train_data, train_label
	
	def get_test_batch(self, st_idx, ed_idx):
		uIds = self.test_uid[st_idx:ed_idx]
		mIds = self.test_mid[st_idx:ed_idx]
		test_data = np.concatenate([self.user_emb[uIds], self.movie_emb[mIds]], axis=1)
		test_label = self.test_binary[st_idx:ed_idx] if self.easy else self.test_rating[st_idx:ed_idx]
		return test_data, test_label

class MLP(kr.Model):
	""" Multi-layer perceptrons """
	def __init__(self, args, data_loader, epoch=5, batch_size=1000, learning_rate=1e-3):
		super().__init__()
		self.flatten = kr.layers.Flatten()
		self.dense1 = kr.layers.Dense(
											units=128,
											activation=tf.nn.relu
											)
		self.dense2 = kr.layers.Dense(
											units=32,
											activation=tf.nn.relu
											)
		self.dense3 = kr.layers.Dense(
											units=2 if args.easy else 10,
											)
		self.optimizer = kr.optimizers.Adam(learning_rate=learning_rate)
		self.epoch = epoch
		self.batch_size = batch_size
		self.data_loader = data_loader

	def call(self, input_data):
		flattened = self.flatten(input_data)
		l1 = self.dense1(flattened)
		l2 = self.dense2(l1)
		l3 = self.dense3(l2)
		output = tf.nn.softmax(l3)
		return output

	def train(self):
		print("**********\nStart to train\n**********")
		start = time.time()
		for i in range(int(self.data_loader.num_train_data / self.batch_size * self.epoch)):
			st = time.time()
			X, y = self.data_loader.get_batch(self.batch_size)
			with tf.GradientTape() as tape:
				pred = self.call(X)
				loss = kr.losses.sparse_categorical_crossentropy(y_true=y, y_pred=pred)  # mean_squared_error
				loss = tf.reduce_mean(loss)
			grads = tape.gradient(loss, self.variables)
			self.optimizer.apply_gradients(grads_and_vars=zip(grads, self.variables))
			ed = time.time()
			if i % 200 == 0:
				print("Batch %d: loss = %f, time = %.3f"%(i, loss.numpy(), ed - st))  # divide by 4 because the rating has been amplified by 2.
		endt = time.time()
		print("Total time for training: %.3f"%(endt - start))

	def eval(self):
		print("**********\nEval start\n**********")
		sca = kr.metrics.SparseCategoricalAccuracy()
		mse = kr.metrics.MeanSquaredError()
		mae = kr.metrics.MeanAbsoluteError()
		for i in range(int(self.data_loader.num_test_data / self.batch_size)):
			start_idx = i * self.batch_size
			end_idx = (i + 1) * self.batch_size
			X, y = self.data_loader.get_test_batch(start_idx, end_idx)
			pred = self.call(X)
			argmax = kr.backend.argmax(pred)
			sca.update_state(y_true = y, y_pred = pred)
			mse.update_state(y_true = y, y_pred = argmax)
			mae.update_state(y_true = y, y_pred = argmax)

		print("MSE = %f\nMAE = %f\nSCA = %f"%(mse.result()/4.0, mae.result()/2.0, sca.result()))


if __name__ == "__main__":

	# parse arguments
	args = args_parser()

	# get index information of users and movies
	num_users, user_base, num_movies, movie_base = getIndInfo(args)
	
	# read embeddings
	user_emb, movie_emb = readEmbeddings(args.dir, args.graphemb, num_users, user_base, num_movies, movie_base)
	if args.setting == 1:
		movie_emb = readTextEmbeddings(args.dir, args.textemb, num_movies, movie_base)
	elif args.setting == 2:
		text_emb = readTextEmbeddings(args.dir, args.textemb, num_movies, movie_base)
		movie_emb = np.concatenate([movie_emb, text_emb], axis=1)

	# build X_train, y_train
	dataloader = DataLoader(args, user_emb, movie_emb, user_base, movie_base)
	model = MLP(args, dataloader, epoch=args.epoch, batch_size=args.batch_size, learning_rate=args.lr)
	model.train()
	model.eval()


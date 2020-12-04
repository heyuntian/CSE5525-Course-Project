from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd
import numpy as np
import json
import ast
from sklearn.model_selection import train_test_split

from utils import Indexer

from tqdm import tqdm


def args_parser():
	parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
	parser.add_argument('--links', required=False, default='data/links.csv',
                        help='movieId mapping')
	parser.add_argument('--metadata', required=False, default='data/movies_metadata.csv',
                        help='movie metadata file')
	parser.add_argument('--credit', required=False, default='data/credits.csv',
                        help='credits file')
	parser.add_argument('--rating', required=False, default='data/ratings.csv',
                        help='ratings file')
	args = parser.parse_args()
	return args

def movieIdIndexing(args):
	"""
	build two mappings from the original tmdbId/movieId in links.csv
	to range(movies.size)
	"""
	links = pd.read_csv(args.links)
	movieIds = links['movieId']
	mvid2mid = dict(zip(movieIds, range(movieIds.size)))
	links['newId'] = links['movieId'].map(mvid2mid)
	tmdbIds = links['tmdbId']
	tmid2mid = dict(zip(tmdbIds, links['newId']))
	print("num_of_movies: %d"%(movieIds.size))
	return mvid2mid, tmid2mid, movieIds.size

def readMovieMetadata(args, tmid2mid):
	df = pd.read_csv(args.metadata, usecols=['genres', 'id', 'overview', 'title'])
	# remove rows with invalid Ids
	df.drop(df[df.id.apply(lambda x: not x.isnumeric())].index, inplace=True)
	df = df.astype({'id': 'int32'})
	df = df.rename(columns={'id': 'tmdbId'})
	# insert a column named 'id'
	df.insert(len(df.columns), 'mId', [tmid2mid[x] for x in df['tmdbId']])

	return df

def readCreditData(args, tmid2mid):  # Redundant code w.r.t. readMovieMetaData
	df = pd.read_csv(args.credit).astype({'id': 'str'})
	df.drop(df[df.id.apply(lambda x: not x.isnumeric())].index, inplace=True)
	df = df.astype({'id': 'int32'})
	df = df.rename(columns={'id': 'tmdbId'})
	# insert a column named 'id'
	df.insert(len(df.columns), 'mId', [tmid2mid[x] for x in df['tmdbId']])

	return df

def readRatingData(args, mvid2mid, id_base):  # Redundant code w.r.t. readMovieMetaData
	df = pd.read_csv(args.rating)
	df.drop(['timestamp'], axis=1, inplace=True)
	# df.drop(df[df.id.apply(lambda x: not x.isnumeric())].index, inplace=True)
	df = df.astype({'movieId': 'int32', 'userId': 'int32', 'rating': 'float32'})
	# insert a column named 'id'
	df.insert(len(df.columns), 'mId', [mvid2mid[x] for x in df['movieId']])
	df.drop(['movieId'], axis=1, inplace=True)

	# re-index the users
	user_values = df.userId.unique()
	num_users = len(user_values)
	user2uId = dict(zip(user_values, range(id_base, id_base+num_users)))
	df['uId'] = df['userId'].map(user2uId)
	df.drop(['userId'], axis=1, inplace=True)

	# add binary scores
	df['binary'] = (df['rating'] > 3.5).astype(int)

	return df, user2uId, num_users

if __name__ == "__main__":
	args = args_parser()

	''' get movie id mappings from links.csv
	mvid2mid: mapping from 'movieId' to range(45843)
	tmid2mid: mapping from 'tmdbId' to range(45843)
	num_movies = 45843 (all movies in links.csv)

	Note that in links.csv, there are missing values of tmdbId
	'''
	mvid2mid, tmid2mid, num_movies = movieIdIndexing(args)  # num_movies=45843
	id_base = num_movies

	''' read metadata from movies_metadata.csv
	Only 45463 movies have valid metadata.
	'''
	movies = readMovieMetadata(args, tmid2mid)
	print("movies.shape %s"%(str(movies.shape)))

	''' create overviews.csv
	contains a header line and 45463 data lines,
	each line includes a mId and its overview (some sentences).
	'''
	movies.to_csv("processed_data/overviews.csv", columns=['mId', 'overview'], index=False)
	movies.to_csv("processed_data/mId2Title.csv", columns=['mId', 'tmdbId', 'title'], index=False)

	''' create genres
	mId2Genre: 45463 lines, each line includes (mId, num of genres, gIds)
	Genre2Id:  20 lines, each line includes (gId, genre name)
	gId ranges from 45843 to 45862
	'''
	f = open("processed_data/mId2Genre.txt", "w")
	genreIdx = Indexer()
	for idx, row in movies.iterrows():
		mId, raw_genres = row['mId'], row['genres']
		raw_genres = raw_genres.replace("\'", "\"")
		genres_l = json.loads(raw_genres)
		f.write("%d %d"%(mId, len(genres_l)))
		for g in genres_l:
			f.write(" %d"%(genreIdx.add_and_get_index(g['name']) + id_base))
		f.write("\n")
	f.close()

	f = open("processed_data/Genre2Id.txt", "w")
	num_genres = len(genreIdx)
	for i in range(num_genres):
		f.write("%d %s\n"%(i + id_base, genreIdx.get_object(i)))
	f.close()
	id_base += num_genres

	''' create credits
	mId2CC.txt: 45476 lines
	each line includes (mId, num of crew/casts, cIds)
	'''
	credits = readCreditData(args, tmid2mid)
	print("credits.shape %s"%(str(credits.shape)))
	cIdx = Indexer()
	f = open("processed_data/mId2CC.txt", "w")
	for idx, row in credits.iterrows():
		mId, raw_cast, raw_crew = row['mId'], row['cast'], row['crew']
		cast_l = ast.literal_eval(raw_cast)
		crew_l = ast.literal_eval(raw_crew)
		attr = []
		for c in crew_l:
			if c['job'].lower() == "director":
				attr.append(cIdx.add_and_get_index(c['name']) + id_base)
		for c in cast_l:
			if int(c['order']) < min(8, len(cast_l)):
				attr.append(cIdx.add_and_get_index(c['name']) + id_base)
		f.write("%d %d"%(mId, len(attr)))
		for att in attr:
			f.write(" %d"%(att))
		f.write("\n")
	f.close()
	num_cast = len(cIdx)
	print("num of cast/crews: %d"%(num_cast))
	id_base += num_cast

	''' create ratings
	train.csv: training data, each line includes <uId, mId, binary rating, rating>
	test.csv: test data, each line includes <uId, mId, binary rating, rating>
	'''
	ratings, user2uId, num_users = readRatingData(args, mvid2mid, id_base)
	X_train, X_test, y_train, y_test = train_test_split(ratings[['uId', 'mId']], ratings[['binary', 'rating']], train_size=0.9)
	train = pd.concat([X_train, y_train], axis=1, sort=False)
	test = pd.concat([X_test, y_test], axis=1, sort=False)
	train.to_csv("processed_data/rating_train.csv", columns=['uId', 'mId', 'binary', 'rating'], index=False)
	test.to_csv("processed_data/rating_test.csv", columns=['uId', 'mId', 'binary', 'rating'], index=False)
	
	print("Finished: \nnum_movies %d \nnum_genres %d \nnum_cast %d \nnum_users %d \n--- \ntotal %d"%(num_movies, num_genres, num_cast, num_users, id_base + num_users))

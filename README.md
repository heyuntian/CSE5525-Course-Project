Here is a slide for this project. [Here][https://heyuntian.github.io/file/CSE_5525_Final_presentation.pdf]

# Requirements

See `requirements.txt`.


# Dataset

The movies dataset from Kaggle: [https://www.kaggle.com/rounakbanik/the-movies-dataset]. Data directory: `./data/`.

## Context

These files contain metadata for all 45,000 movies listed in the Full MovieLens Dataset. The dataset consists of movies released on or before July 2017. Data points include cast, crew, plot keywords, budget, revenue, posters, release dates, languages, production companies, countries, TMDB vote counts and vote averages.

This dataset also has files containing 26 million ratings from 270,000 users for all 45,000 movies. Ratings are on a scale of 0.5-5 and have been obtained from the official GroupLens website.

## Content

  - **movies_metadata.csv**: The main Movies Metadata file. Contains information on 45,000 movies featured in the Full MovieLens dataset. Features include posters, backdrops, budget, revenue, release dates, languages, production countries and companies. *This file contains tmdbId and imdbId, while only tmdbId is included in all records*.

  - **credits.csv**: Consists of Cast and Crew Information for all our movies. Available in the form of a stringified JSON Object. *This file uses tmdbId as the index.*

  - **links.csv**: The file that contains the TMDB and IMDB IDs of all the movies featured in the Full MovieLens dataset. *This file has movieId, tmdbId, and imdbIds, all of them are not consecutive. The processed data has assign consecutive mIds for movies and provides mappings to tmdbId and movieId.*

  - **ratings.csv**: This file contains 26 million ratings from 2700 users on all 45,000 movies.


# Data processing

**preprocessing.py** creates relationship and mappings based on the raw dataset. It has the following arguments:

  - `--links`: the path to links.csv, default is `data/links.csv`
  - `--metadata`: the path to the movies' metadata file, default is `data/movies_metadata.csv`
  - `--credit`: the path to credit data, default is `data/credits.csv`
  - `--rating`: the path to rating data, default is `data/ratings.csv`

Running `preprocessing.py` will build the following files in the `./processed_data/` directory.

  - `Genre2Id.txt`: 20 lines, each line has <gId, genre>.
  - `mId2CC.txt`: each line has <mId, # of cast/crews, cIds> representing each movie's director(s) and top 8 casts.
  - `mId2Genre.txt`: each line has <mId, # of genres, gIds> representing each movie's genre attributes.
  - `mId2Title.csv`: each line consists of <mId, tmdbId, title>.
  - `overviews.csv`: each line has <mId, overview>.
  - `rating_test.csv`: each line has <uId, mId, binary, rating> representing a user, a movie, if the user's rating on that movie > 3.5 (50%), and its exact rating (0.5 - 5.0). 
  - `rating_train.csv`: similar to the test data.

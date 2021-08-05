import json
import os
from tqdm import tqdm
import pandas as pd
import pickle as pkl
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents, Lowercase, Strip
from ast import literal_eval
import numpy as np
import random
import pickle
from transformers import BertTokenizerFast


reviews = open("data/mard/mard_reviews.json")
metaData = open("data/mard/mard_metadata.json")
ACCOUSTIC_BRAINZ_FOLDER = "data/mard/Augmented"
PROCESSED_DATA_FOLDER = "pre_processed_data"
SENTLENGTH = 400



def load_accoustic_data(reviews):
	
	test = []
	train = []
	dev  = []
	files =os.listdir(ACCOUSTIC_BRAINZ_FOLDER)
	random.shuffle(files)
	for idx, filename in tqdm(enumerate(files)):
		# if idx>500:
			# break
		feat = {}
		feat['album']= ""
		feat['artist']= ""

		file = ACCOUSTIC_BRAINZ_FOLDER +"/" + filename
		amazonID = filename[:filename.find("_")]
		descriptor = open(file).read()
		features = json.loads(descriptor)
		reviewsHere = [ (idx,x) for  idx, x in enumerate(reviews) if x['amazon-id'] == amazonID ]

		for review in reviewsHere:
			feat = {}
			feat['album']= ""
			feat['artist']= ""
			feat["amazon_id"] = amazonID
			if 'album' in features['metadata']['tags']:
				feat['album'] = features['metadata']['tags']['album']
			if 'artist' in features['metadata']['tags']:
				feat['artist'] = features['metadata']['tags']['artist']
			feat['review_text'] = review[1]['reviewText']
			feat['reviews'] = tokenized_corpus[review[0]].ids
			feat['review_id'] = review[0]

			if len(feat['reviews']) > SENTLENGTH:
				feat['reviews'] = feat['reviews'][:SENTLENGTH]
			if len(feat['reviews'])<SENTLENGTH:
				feat['reviews'] = [30000 for x in range(SENTLENGTH - len(feat['reviews']))] + feat['reviews']
			if 'genre' in features['metadata']['tags']:
				feat['genre'] = features['metadata']['tags']['genre'][0]
			if 'musicbrainz_recordingid' in features['metadata']['tags']:
				feat['id'] = features['metadata']['tags']['musicbrainz_recordingid'][0]
			feat['mfcc_cov'] = features['lowlevel']['mfcc']['cov']
			feat['mfcc_icov'] = features['lowlevel']['mfcc']['icov']
			feat['mfcc_mean'] = features['lowlevel']['mfcc']['mean']
			zcross_rate = features['lowlevel']['zerocrossingrate']
			feat['zero_crossing_rate'] = [zcross_rate['min'], zcross_rate['max'], zcross_rate['dvar2'], zcross_rate['median'], zcross_rate['dmean2'], zcross_rate['dmean'], zcross_rate['var'], zcross_rate['dvar'], zcross_rate['mean']]
			rolloff = features['lowlevel']['spectral_rolloff']
			feat['spectral_rolloff'] = [rolloff['min'], rolloff['max'], rolloff['dvar2'], rolloff['median'], rolloff['dmean2'], rolloff['dmean'], rolloff['var'], rolloff['dvar'], rolloff['mean']]


			# feats.append(feat)
			if idx < 2600:
				test.append(feat)
			elif idx< 5260:
				dev.append(feat)
			else:
				train.append(feat)


	test = pd.DataFrame(test)
	test.to_csv(PROCESSED_DATA_FOLDER + "/test-full-blind.csv")

	dev = pd.DataFrame(dev)
	dev.to_csv(PROCESSED_DATA_FOLDER + "/dev-full-blind.csv")

	train = pd.DataFrame(train)
	train.to_csv(PROCESSED_DATA_FOLDER + '/train-full-blind.csv')

		# print(reviewsHere)
	return True



def loadTokenizer(vocab_size):
	tokenizer_file_path = PROCESSED_DATA_FOLDER + "/tokenizer.json"

	if os.path.exists(tokenizer_file_path):
		print("Loading Tokenizer from file")
		tokenizer = Tokenizer.from_file(tokenizer_file_path)
	else:
		print("Generating Tokenizer")
		tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
		normalizer = normalizers.Sequence([ Strip()])
		tokenizer.normalizer = normalizer
		trainer = WordLevelTrainer( vocab_size =vocab_size, special_tokens=["[UNK]"])
		tokenizer.pre_tokenizer = Whitespace()
		corpus =[]
		with open(PROCESSED_DATA_FOLDER + "/corpus-trim.txt", "rt") as f:
			for idx, line in enumerate(f):
				corpus.append(line)
		tokenizer.train_from_iterator(corpus, trainer)
		tokenizer.save(tokenizer_file_path)

	return tokenizer

def loadTokenizedCorpus(tokenizer):
	pklfilepath = PROCESSED_DATA_FOLDER + "/tokenized_corpus.pkl"

	if os.path.exists(pklfilepath):
		print("Loading Tokenized Corpus")

		with open(pklfilepath, "rb") as f:
			tokenized_corpus = pkl.load(f)
	else:
		print("Tokenizing Corpus")
		corpus=[]
		with open(PROCESSED_DATA_FOLDER +"/corpus-trim.txt", "rt") as f:
			for idx, line in tqdm(enumerate(f)):

				corpus.append(line)
		# tokenizer.enable_padding(direction="left", length =400)
		tokenized_corpus = tokenizer.encode_batch(corpus)
		with open(pklfilepath, "wb") as f:
			pkl.dump(tokenized_corpus, f)
	return tokenized_corpus



def load_reviews():
	reviewObj = []
	corpus = []
	reviews = open("data/mard/mard_reviews.json")
	reviewLgth = 0
	count = 0
	# corpus = ""

	for review in tqdm(reviews.readlines()):
		review = json.loads(review)
		reviewObj.append(review)
		reviewLgth += len(review['reviewText'])
		# corpus.append(review['reviewText'])
		count+=1
	print("Average Review Length: " + str(reviewLgth/count))
	return reviewObj



def main ():
	reviews = load_reviews()
	tokenizer = loadTokenizer(30000)
	tokenized_corpus = loadTokenizedCorpus( tokenizer)
	load_accoustic_data(reviews)

def get_features(data):
		print("Loading Reviews...")
		data['reviews'] = data['reviews'].apply(literal_eval)
		data['reviews'] = data['reviews'].apply(lambda x: np.array(x))

		print("Loading MFCC Cov...")
		data['mfcc_cov'] = data['mfcc_cov'].apply(literal_eval)
		data['mfcc_cov'] = data['mfcc_cov'].apply(lambda x: np.array(x))

		print("Loading MFCC ICov...")
		data['mfcc_icov'] = data['mfcc_icov'].apply(literal_eval)
		data['mfcc_icov'] = data['mfcc_icov'].apply(lambda x: np.array(x))

		print("Loading MFCC MEan...")
		data['mfcc_mean'] = data['mfcc_mean'].apply(literal_eval)
		data['mfcc_mean'] = data['mfcc_mean'].apply(lambda x: np.array(x))

		print("Loading zerocrossingrate...")
		data['zero_crossing_rate'] = data['zero_crossing_rate'].apply(literal_eval)
		print("Loading spectral rolloff...")
		data['spectral_rolloff'] = data['spectral_rolloff'].apply(literal_eval)


		text = np.stack(data['reviews'].to_numpy())
		samples = text.shape[0]

		mfcc_cov = np.stack(data['mfcc_cov'].to_numpy()).reshape(samples, 13 * 13)
		mfcc_icov = np.stack(data['mfcc_icov'].to_numpy()).reshape(samples, 13 * 13)
		mfcc_mean = np.stack(data['mfcc_mean'].to_numpy()).reshape(samples, 13 )
		spectral_rolloff = np.stack(data['spectral_rolloff'].to_numpy()).reshape(samples, 9)
		zero_crossing_rate = np.stack(data['zero_crossing_rate'].to_numpy()).reshape(samples, 9)


		audio = np.append(mfcc_cov, mfcc_icov, axis=1)
		audio = np.append(audio, mfcc_mean, axis=1)
		audio = np.append(audio, spectral_rolloff, axis=1)
		audio = np.append(audio, zero_crossing_rate, axis=1)

		features = np.append(text, audio, axis =1)
		return features

def pull_train_data():
	train_file_path = PROCESSED_DATA_FOLDER + "/train.npy"
	tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

	if os.path.exists(train_file_path):
		print("Loading Training Data from file")

		features  = np.load(train_file_path, allow_pickle= True)
		print("Loaded " + str(features.shape[0]) + " Train Reviews" )
		lookupSongs = open("pre_processed_data/lookupSongs.pkl", "rb")
		lookupSongs = pickle.load(lookupSongs)

		lookupReviews = open("pre_processed_data/lookupReviews.pkl", "rb")
		lookupReviews = pickle.load(lookupReviews)

	else:
		print("Saving Train Data to file")

		amz2Idx = {}
		idx2Amz = {}
		
		lookupReviews = {}
		lookupSongs = {}

		data = pd.read_csv(PROCESSED_DATA_FOLDER + "/train-full-blind.csv")


		print("Computing Lookup Tables")
		count = 0
		for index, row in tqdm(data.iterrows()):
			dataOver = data[data['amazon_id'] == row['amazon_id']]
			lookupReviews[index] = dataOver.index[dataOver['review_id'] != row['review_id']].tolist()

			lookupSongs[index] = data.index[data['review_id'] == row['review_id']].tolist()
		
		
		features = get_features(data)

		f = open(PROCESSED_DATA_FOLDER + "/lookupSongs.pkl","wb")
		pickle.dump(lookupSongs,f)
		f.close()

		f = open(PROCESSED_DATA_FOLDER + "/lookupReviews.pkl","wb")
		pickle.dump(lookupReviews,f)
		f.close()

		np.save(train_file_path, features, allow_pickle = True)
		print("Loaded " + str(features.shape[0]) + " Train Reviews" )
	return  features, lookupReviews, lookupSongs


def pull_dev_data():
	dev_file_path = PROCESSED_DATA_FOLDER + "/dev.npy"
	# tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

	if os.path.exists(dev_file_path):
		features  = np.load(dev_file_path, allow_pickle= True)

	else:
		print("Saving Dev Data to file")
		data = pd.read_csv(PROCESSED_DATA_FOLDER + "/dev-full-blind.csv")

		data = data.dropna()



		features = get_features(data)
		print(features.shape)

		np.save(dev_file_path, features, allow_pickle = True)
		print("Loaded " + str(features.shape[0]) + " Dev Reviews" )
	return features
	

def pull_artists():
	PROCESSED_DATA_FOLDER = "pre_processed_data"

	data = pd.read_csv(PROCESSED_DATA_FOLDER + "/featurs.csv")

	artists = data['artist'].to_numpy()


	return artists

def pull_album():
	PROCESSED_DATA_FOLDER = "pre_processed_data"

	data = pd.read_csv(PROCESSED_DATA_FOLDER + "/featurs.csv")

	album = data['album'].to_numpy()


	return album

def pull_reviews():
	PROCESSED_DATA_FOLDER = "pre_processed_data"

	data = pd.read_csv(PROCESSED_DATA_FOLDER + "/featurs.csv")

	artists = data['review_text'].to_numpy()


	return artists

if __name__ == "__main__":
	reviews = load_reviews()
	tokenizer = loadTokenizer(30000)
	tokenized_corpus = loadTokenizedCorpus( tokenizer)
	load_accoustic_data(reviews)

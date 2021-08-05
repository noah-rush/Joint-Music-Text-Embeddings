from Training import Trainer
import Data
from Model import TwoBranch
from Evaluation import evaluate
from Evaluation import randomBaseline

import torch
import numpy as np
from argparse import ArgumentParser

SENTLENGTH = 400

parser = ArgumentParser()
parser.add_argument("--mode", type=str, default=False, required=True)
parser.add_argument("--loadepoch", type=int, default=False)
parser.add_argument("--recallk", type=int, default=False)
parser.add_argument("--neighborhood", type=int, default=0)
parser.add_argument("--bidirectional", type=int, default=1)


args = parser.parse_args()


BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 200

print("Loading Data")
TRAIN, lookupReviews, lookupAlbums = Data.pull_train_data()
DEV = Data.pull_dev_data()

TEXT_EMBED_DIM = 128
FINAL_EMBED_DIM = 128 
FC_DIM = 256
VOCAB_SIZE = 30000
MARGIN = 0.05
CHECKPOINT = 10
K = 10
AUDIOFEATURES = 169
NEIGHBORSAMPLE = False
if args.neighborhood ==1:
	NEIGHBORSAMPLE = True
BIDIR = args.bidirectional



print("Initiating Model")
MODEL = TwoBranch(AUDIOFEATURES, FC_DIM, TEXT_EMBED_DIM, FINAL_EMBED_DIM, VOCAB_SIZE)


if args.mode == "Train":
	print("training")
	trainer = Trainer(EPOCHS, LEARNING_RATE, BATCH_SIZE,TRAIN, DEV,  MODEL, MARGIN, CHECKPOINT, K, lookupAlbums, lookupReviews, NEIGHBORSAMPLE, BIDIR)
	trainer.train()

if args.mode == "Dev":
	ALBUMS = Data.pull_album()
	MODEL.load_state_dict(torch.load("chkpoints/model" + str(args.loadepoch)+ ".pth"))
	MODEL.cuda()
	evaluate(DEV, MODEL, ALBUMS, args.recallk)

if args.mode == "Random":
	randomBaseline()





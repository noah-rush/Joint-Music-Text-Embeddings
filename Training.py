import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import Data
import random
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
from Evaluation import evaluate
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

# writer = SummaryWriter('runs/FirstDraft')

from transformers import DistilBertTokenizer, DistilBertModel





class Trainer():
	def __init__(self, EPOCHS, LR, BATCH_SIZE, TRAIN,  DEV, MODEL, MARGIN, CHKPOINT, K, lookupAlbums, lookupReviews, NEIGHBORSAMPLE, BIDIR):
		self.chkpoint = CHKPOINT
		self.epochs = EPOCHS
		self.lr = LR
		self.batch_size = BATCH_SIZE
		self.data = torch.from_numpy(TRAIN)
		self.train_samples = TRAIN.shape[0]
		self.lookupAlbums = lookupAlbums
		self.lookupReviews = lookupReviews
		trainData = TensorDataset(torch.from_numpy(TRAIN), torch.arange(self.train_samples))
		self.data_loader = DataLoader(trainData, shuffle =True, batch_size = self.batch_size)
		self.model = MODEL.cuda()
		self.optim = torch.optim.Adam(MODEL.parameters(), lr=LR)
		self.criterion = torch.nn.MarginRankingLoss(MARGIN)
		self.margin = MARGIN
		self.ksamples = K
		self.DEV = DEV
		self.neighbor = NEIGHBORSAMPLE
		self.bidir = BIDIR

	def dist(self,x, y):
		return torch.sqrt(torch.sum(torch.square(x - y), dim=1))

	def generateVisualization(self,epoch):
		ARTISTS = Data.pull_artists()
		REVIEWS = Data.pull_reviews()

		n = 500
		perm = torch.randperm(len(self.data))
		reviews = REVIEWS[perm][:n]
		artists = ARTISTS[perm][:n]
		data = self.data[perm][:n]

		print(artists[:5])
		self.model.eval()
		text = torch.from_numpy(data[:,:400]).long().cuda()
		audio = torch.from_numpy(data[:,400:]).float().cuda()

		textEmbed, audEmbed = self.model(text, audio)
		closest = torch.argmin(torch.torch.cdist(textEmbed, audEmbed), dim=1)
		print(reviews[closest[:5].cpu()])
		
		allEmbeds = torch.cat((textEmbed, audEmbed), dim = 0)
		meta = np.append(reviews, artists, axis=0)
		writer.add_embedding(allEmbeds, metadata= meta, global_step = epoch)
		self.model.train()

	def validate(self):
		ALBUMS = Data.pull_album()
		self.model.eval()
		evaluate(self.DEV, self.model, ALBUMS)
		self.model.train()

	def train(self):
		self.validate()
		for epoch in tqdm(range(self.epochs)):
			loss_val = 0
			for batch, idxs in tqdm(self.data_loader):
				
				self.optim.zero_grad()

				items = batch.shape[0]
				text = batch[:,:400].long().cuda()
				audio = batch[:,400:400+169].float().cuda()
				

				origText = torch.clone(text)
				origAudio = torch.clone(audio)

				if self.neighbor:

					homeText = torch.zeros((1,400)).cuda()
					neighborTexts = torch.zeros((1,400)).cuda()

					homeAudio = torch.zeros((1,169)).cuda()
					neighborAudio = torch.zeros((1,169)).cuda()

					# Neighborhood Sampling
					for idx, id_num in enumerate(idxs):
						id_num = id_num.item()
						if self.lookupReviews[id_num]!=  []:
							homeText = torch.cat((homeText, origText[idx].unsqueeze(0) ))
							neighborTexts = torch.cat((neighborTexts, self.data[random.choice(self.lookupReviews[id_num])].unsqueeze(0)[:,:400].cuda().long() ))
						if self.lookupAlbums[id_num] != []:
							homeAudio = torch.cat((homeAudio, origAudio[idx].unsqueeze(0) ))
							neighborAudio = torch.cat((neighborAudio, self.data[random.choice(self.lookupAlbums[id_num])].unsqueeze(0)[:,400:400+169].cuda().float() ))
					
					neighborTextItems = homeText.size()[0] - 1
					neighborAudioItems = homeAudio.size()[0] - 1

					homeText = homeText[1:].long()
					neighborTexts = neighborTexts[1:].long()
					homeAudio = homeAudio[1:]
					neighborAudio = neighborAudio[1:]

					# print(homeText)

					homeTextEmbed, homeAudEmbed = self.model(homeText, homeAudio)
					neighborTextsEmbed, neighborAudioEmbed = self.model(neighborTexts, neighborAudio)

					trueDistsText = self.dist(homeTextEmbed ,neighborTextsEmbed)
					trueDistsAudio = self.dist(homeAudEmbed , neighborAudioEmbed)

					for k in range(5):
						randSample = torch.randperm(items)[:neighborTextItems]
						fakeText = origText[randSample]
						randSample = torch.randperm(items)[:neighborAudioItems]
						fakeAudio = origAudio[randSample]
						fakeText, fakeAudio = self.model(fakeText, fakeAudio)
						fakeDistsText = self.dist(homeTextEmbed, fakeText)
						fakeDistsAudio = self.dist(homeAudEmbed, fakeAudio)
						if k == 0:
							loss = self.criterion(trueDistsText, fakeDistsText, torch.ones(neighborTextItems).cuda() * -1)
							loss += self.criterion(trueDistsAudio, fakeDistsAudio, torch.ones(neighborAudioItems).cuda() * -1)
						else:
							loss += self.criterion(trueDistsText, fakeDistsText, torch.ones(neighborTextItems).cuda() * -1)
							loss += self.criterion(trueDistsAudio, fakeDistsAudio, torch.ones(neighborAudioItems).cuda() * -1)

					


				for k in range(self.ksamples):
					randSample = torch.randperm(items)
					text= torch.cat((text, origText))
					audio = torch.cat((audio, origAudio[randSample]))
					if self.bidir:
						text = torch.cat((text, origText[randSample]))
						audio = torch.cat((audio, origAudio))

				origText = torch.clone(text)
				origAudio = torch.clone(audio)


				textEmbed, audEmbed = self.model(text, audio)

				trueDists = torch.sqrt(torch.sum(torch.square(textEmbed[:items] - audEmbed[:items]), dim=1))
				
				modder = 1
				if self.bidir:
					modder = 2
				for i in range(self.ksamples * modder):
					fakeDists = torch.sqrt(torch.sum(torch.square(textEmbed[items * (i +1) :items * (i +1) + items] - audEmbed[items * (i +1) :items * (i +1) + items]), dim=1))
					if i == 0 or not self.neighbor:
						loss = self.criterion(trueDists, fakeDists, torch.ones(items).cuda() * -1)
					else:
						loss+= self.criterion(trueDists, fakeDists, torch.ones(items).cuda() * -1)
				

				loss_val += loss.item()/ (self.ksamples/2)
				loss.backward()
				self.optim.step()

			print("Epoch Loss: " + str(loss_val))
			if epoch % self.chkpoint == 0:
				torch.save(self.model.state_dict(), "chkpoints/model" + str(epoch)+ ".pth")
				self.generateVisualization(epoch)
				self.validate()
		writer.close()

		





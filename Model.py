import torch.nn as nn
import torch.nn.functional as F
import torch

class TwoBranch(nn.Module):
	def __init__(self, audioFeatures, hiddenDim, textEmbedDim, finalEmbedDim, vocab_size):
		super(TwoBranch, self).__init__()
		# print(vocab_size+1)
		self.embeddings = nn.Embedding(vocab_size + 1, textEmbedDim, padding_idx =vocab_size)
		# self.textRNN = nn.LSTM(textEmbedDim, int(textEmbedDim/2), bidirectional= True, batch_first = True, num_layers = 1)
		self.audioBranch = nn.Sequential(
			nn.Linear(audioFeatures,  hiddenDim),
			nn.ReLU(),
			nn.Linear(hiddenDim, finalEmbedDim)

		)
		self.textBranch = nn.Sequential(
			nn.Linear(textEmbedDim ,  hiddenDim),
			nn.ReLU(),
			nn.Linear(hiddenDim, finalEmbedDim)
		)
	def forward(self, text, audio):
		# print(text.shape)
		# print(text)
		text = self.embeddings(text)
		# print(text.size())
		# textOut, hidden = self.textRNN(text)
		# print(hidden[0].size())
		# del textOut
		# textOut = hidden[0].permute(1,0,2)
		# textOut = torch.cat((textOut[:,0], textOut[:,1]), dim =1)
		# print(textOut.size())

		textOut = torch.mean(text, dim=1)
		textOut = self.textBranch(textOut)
		# print(textOut.size())
		textOut = F.normalize(textOut)

		audioOut = self.audioBranch(audio)
		audioOut = F.normalize(audioOut)

		return textOut, audioOut


class EncoderLSTM(nn.Module):
	def __init__(self, input_dim, hidden_dim):
		super(TwoBranch, self).__init__()
		self.rnn = nn.LSTM(input_dim, hidden_dim, bidirectional=True)
	def forward(self, inputs):
		outputs, (hidden, cell) = self.rnn(inputs)
		return outputs, hidden, cell

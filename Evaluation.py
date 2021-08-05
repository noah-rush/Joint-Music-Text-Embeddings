import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

def evaluate(dev, model, ALBUMS):
	model.eval()
	loader = DataLoader(torch.from_numpy(dev), shuffle = False, batch_size = 256)
	
	for idx, batch in enumerate(loader):
		text = batch[:,:400].long().cuda()
		audio = batch[:,400:400+169].float().cuda()
		# print("evaluation script")
		textEmbed, audEmbed = model(text, audio)
		if idx == 0:
			textResults = textEmbed.detach().cpu()
			audResults = audEmbed.detach().cpu()
		else:
			textResults = torch.cat((textResults, textEmbed.detach().cpu()))
			audResults = torch.cat((audResults, audEmbed.detach().cpu()))

	print(audResults.size())
	results = torch.cdist(textResults, audResults)

	res, ind = torch.topk(results, k=1, dim = 1, largest = False)
	
	w = torch.arange(ind.size()[0]).unsqueeze(1).cuda()
	correct = 0

	for idx, result in enumerate(ind):
		album = ALBUMS[idx]
		# print(album)
		resultAlbums = [ALBUMS[x] for x in result]
		# print(resultAlbums)
		if album in resultAlbums:
			correct += 1
	print("__________________________________________")
	
	print("Review-to-Audio Recall@1: " + str(correct / ind.size()[0]))

	res, ind = torch.topk(results, k=5, dim = 1, largest = False)
	
	w = torch.arange(ind.size()[0]).unsqueeze(1).cuda()
	correct = 0

	for idx, result in enumerate(ind):
		album = ALBUMS[idx]
		# print(album)
		resultAlbums = [ALBUMS[x] for x in result]
		# print(resultAlbums)
		if album in resultAlbums:
			correct += 1

	print("Review-to-Audio Recall@5: " + str(correct / ind.size()[0]))

	res, ind = torch.topk(results, k=10, dim = 1, largest = False)
	
	w = torch.arange(ind.size()[0]).unsqueeze(1).cuda()
	correct = 0

	for idx, result in enumerate(ind):
		album = ALBUMS[idx]
		# print(album)
		resultAlbums = [ALBUMS[x] for x in result]
		# print(resultAlbums)
		if album in resultAlbums:
			correct += 1

	print("Review-to-Audio Recall@10: " + str(correct / ind.size()[0]))
	print("__________________________________________")


def randomBaseline():

	ind = torch.randint(0,6597, size = (6597, 1))
	
	correct = 0

	for idx, result in tqdm(enumerate(ind)):
		if idx in result:
			correct += 1

	print(correct / ind.size()[0])





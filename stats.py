import Data
import torch
import numpy as np
from tqdm import tqdm

albums  = Data.pull_album()
reviews  = Data.pull_reviews()

print(albums)
allAlbums = {}
for album in tqdm(albums):
	if album in allAlbums:
		allAlbums[album] += 1
	else:
		allAlbums[album] = 1


albumsSorted = dict(sorted(allAlbums.items(), key=lambda item: item[1]))
print(len(allAlbums))
print(albumsSorted)
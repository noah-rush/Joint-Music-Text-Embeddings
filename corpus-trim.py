PROCESSED_DATA_FOLDER = "pre_processed_data"
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english')) 
from tqdm import tqdm

corpus=[]
with open(PROCESSED_DATA_FOLDER +"/corpus.txt", "rt") as f:
	for idx, line in tqdm(enumerate(f)):
		line = line.split()
		line = [word for word in line if not word in stop_words]
		# print(line)
		line = " ".join(line)
		corpus.append(line)

trim_corp = open(PROCESSED_DATA_FOLDER + "/corpus-trim.txt", 'w')
for line in corpus:
	trim_corp.write(line + "\n")
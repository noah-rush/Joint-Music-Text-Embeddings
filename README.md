# Music-Text-Embeddings

## About
Music-Text Embeddings is a final project from CS 5787 -Deep Learning at Cornell Tech. In this project we built a two branch neural network to correlate text reviews with low-level audio features. The data was taken from the Multimodal Album Review dataset (MARD), available (https://www.upf.edu/web/mtg/mard). The dataset matches amazon reviews with audio features stored in the AcousticBrainz database. AcousticBrainz stores audio features for songs, including low-level features like spectral rolloff, and high-level features like danceability. We use a two-branch neural network, that takes in audio features and text features, and attempts to learn weights that will map them into a shared embedding space. We use a ranking loss which forces close pairs of audio and text to map close together, and false pairs to spread apart. We evaluate our results by holding out audio and text samples, and then embedding them in the learned space, and evaluating recall to the corresponding album or text pair. For the full paper with our results and various experiments, click here (https://www.noahrush.com/public/pdfs/DL_Project.pdf).

## Implementation
We use PyTorch to build and run the model, with Tensorboard used to visualize the model results and embedding space. We use DistilBert from hugging face for our BERT evaluation. (https://huggingface.co/transformers/model_doc/distilbert.html)

## Usage
Download the mard dataset into a data file within thre repo and run `python Data.py`. This will generate dev,test, and train CSVs with audio features pulled out and reviews converted into vocab identifiers. From here you can run `python main.py --mode Train`. This will train the model without neighborhood sampling and with a bidirectional ranking-loss. Flags for changing those two options are `--neighbor true` and `bidirectional 0`. Standard options for Neural network training are in main.py. (Batch size, Learning rate, checkpoint, etc..) The model is built to run on a GPU. Every checkpoint during train, the model will evaluate its performance on the dev set.



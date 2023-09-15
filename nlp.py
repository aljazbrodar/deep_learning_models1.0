import nltk 
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, gutenberg
from nltk.stem.porter import *
nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('stopwords')

import string 

import gensim
from gensim.models.phrases import Phraser, Phrases
from gensim.models.word2vec import Word2Vec
    
from sklearn.manifold import TSNE

import pandas as pd 
from bokeh.io import output_notebook, output_file
from bokeh.plotting import show, figure
import matplotlib.pyplot as plt

gberg_sents = gutenberg.sents()

stpwrds = stopwords.words('english') + list(string.punctuation)

stemmer = PorterStemmer()

lower_sents = []
for s in gberg_sents:
    lower_sents.append([w.lower() for w in s if w.lower()
                        not in list(string.punctuation)])
    


lower_bigram = Phraser(Phrases(lower_sents, min_count=32, threshold=64))

clean_sets = []

for s in lower_sents:
    clean_sets.append(lower_bigram[s])
    
model =     Word2Vec(sentences=clean_sets, vector_size=64, sg=1, window=10, epochs=5, min_count=10, workers=4)

model.save('clean_gutenberg_model.w2v')

model = gensim.models.Word2Vec.load('clean_gutenberg_model.w2v')

tsne = TSNE(n_components=2, n_iter=1000)
word_vectors = model.wv[model.wv.index_to_key]  # Use model.wv.index2word to get the list of words
X_2d = tsne.fit_transform(word_vectors)
coords_df = pd.DataFrame(X_2d, columns=['x', 'y'])
coords_df['token'] = model.wv.index_to_key

_= coords_df.plot.scatter('x','y', figsize=(12,12), marker='.', s=10, alpha=0.2)

output_notebook()
subset_df = coords_df.sample(n=5000)
p = figure(width=400, height=400)
_=p.text(x=subset_df.x, y=subset_df.y, text=subset_df.token)
output_file("bokeh_plot.html")
show(p)


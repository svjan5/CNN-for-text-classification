
# coding: utf-8

# In[7]:

import numpy as np
import pandas as pd

from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import SnowballStemmer
from w2v import train_word2vec 
import pickle, datetime
import difflib

import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.utils import np_utils
from keras.layers import Input, Dense, Dropout, Flatten, Embedding, regularizers
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.merge import Concatenate
from keras import optimizers
from keras.preprocessing import sequence
from keras import backend as K

np.random.seed(0)


# In[8]:

df = pd.read_csv('../../Datasets/SST1_dataset/Processed_SST1.tsv', sep='\t')

raw_docs_train      = df[df.split_ind == 1]['Phrases']
sentiment_train     = df[df.split_ind == 1]['Label']
raw_docs_test       = df[df.split_ind == 2]['Phrases']
sentiment_test      = df[df.split_ind == 2]['Label']
num_labels          = len(np.unique(sentiment_train))


# In[9]:

#text pre-processing
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
stemmer = SnowballStemmer('english')

print ("pre-processing train docs...")
processed_docs_train = []
for doc in raw_docs_train:
   tokens = word_tokenize(doc)
   filtered = [word for word in tokens if word not in stop_words]
   stemmed = [stemmer.stem(word) for word in filtered]
   processed_docs_train.append(stemmed)

print ("pre-processing test docs...")
processed_docs_test = []
for doc in raw_docs_test:
   tokens = word_tokenize(doc)
   filtered = [word for word in tokens if word not in stop_words]
   stemmed = [stemmer.stem(word) for word in filtered]
   processed_docs_test.append(stemmed)
    
processed_docs_all = np.concatenate((processed_docs_train, processed_docs_test), axis=0)

dictionary = corpora.Dictionary(processed_docs_all)
dictionary_size = len(dictionary.keys())

print ("converting to token ids...")
word_id_train, word_id_len = [], []
for doc in processed_docs_train:
    word_ids = [dictionary.token2id[word] for word in doc]
    word_id_train.append(word_ids)
    word_id_len.append(len(word_ids))

word_id_test, word_ids = [], []
for doc in processed_docs_test:
    word_ids = [dictionary.token2id[word] for word in doc]
    word_id_test.append(word_ids)
    word_id_len.append(len(word_ids))
        
seq_len = np.round((np.mean(word_id_len) + 2*np.std(word_id_len))).astype(int)

#pad sequences
word_id_train = sequence.pad_sequences(np.array(word_id_train), maxlen=seq_len)
word_id_test = sequence.pad_sequences(np.array(word_id_test), maxlen=seq_len)
y_train_enc = np_utils.to_categorical(sentiment_train, num_labels)
y_test_enc = np_utils.to_categorical(sentiment_test, num_labels)


# In[19]:

print(word_id_train.shape, word_id_test.shape, y_train_enc.shape, y_test_enc.shape)


# # -------------------------------- Training model  -----------------------------------

# ### Model Paremeters

# In[11]:

model_type    = 'CNN-rand'  # CNN-rand|CNN-non-static|CNN-static
embedding_dim = 25         # word2vec dim
vocab_size    = dictionary_size
max_sent_len  = seq_len
N_CATEGORY    = 5


# ### Generate word2vec 

# In[12]:

if model_type in ['CNN-non-static', 'CNN-static']:
    embedding_wts = train_word2vec( np.vstack((x_train, x_test, x_valid)), 
                                    ind_to_wrd, num_features = embedding_dim)
    if model_type == 'CNN-static':
        x_train = embedding_wts[0][x_train]
        x_test  = embedding_wts[0][x_test]
#         x_valid = embedding_wts[0][x_valid]
        
elif model_type == 'CNN-rand':
    embedding_wts = None
    
else:
    raise ValueError("Unknown model type")


# ### Create model

# In[13]:

batch_size   = 50
filter_sizes = [3,4,5]
num_filters  = 50
dropout_prob = (0.5, 0.8)
hidden_dims  = 50

l2_reg = 0.3

# Deciding dimension of input based on the model
input_shape = (max_sent_len, embedding_dim) if model_type == "CNN-static" else (max_sent_len,)
model_input = Input(shape = input_shape)

# Static model do not have embedding layer
if model_type == "CNN-static":
    z = Dropout(dropout_prob[0])(model_input)
else:
    z = Embedding(vocab_size, embedding_dim, input_length = max_sent_len, name="embedding", trainable=True)(model_input)
    z = Dropout(dropout_prob[0])(z)

# Convolution layers
z1 = Conv1D(    filters=num_filters, kernel_size=filter_sizes[0], 
                padding="valid", activation="relu", 
                strides=1)(z)
z1 = MaxPooling1D(pool_size =2)(z1)
z1 = Flatten()(z1)

z2 = Conv1D(    filters=num_filters, kernel_size=filter_sizes[1], 
                padding="valid", activation="relu", 
                strides=1)(z)
z2 = MaxPooling1D(pool_size=2)(z2)
z2 = Flatten()(z2)

z3 = Conv1D(    filters=num_filters, kernel_size=filter_sizes[2], 
                padding="valid", activation="relu",
                strides=1)(z)
z3 = MaxPooling1D(pool_size=2)(z3)
z3 = Flatten()(z3)

# Concatenate the output of all convolution layers
z = Concatenate()([z1, z2, z3])
z = Dropout(dropout_prob[1])(z)

z = Dense(hidden_dims, activation="relu")(z)
model_output = Dense(N_CATEGORY, activation="softmax")(z)
    
model = Model(model_input, model_output)
model.summary()


# ### Train model 

# In[ ]:

if model_type == "CNN-non-static":
    embedding_layer = model.get_layer("embedding")
    embedding_layer.set_weights(embedding_wts)

model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(0.1), metrics=["accuracy"])

res = model.fit(word_id_train, y_train_enc, 
          batch_size = batch_size,
          epochs=100,
          validation_data=(word_id_test, y_test_enc), verbose=2)


# In[22]:

import os
os.system('say done')


# ### Computing Accuracy

# In[17]:

# Training Accuracy
predictions = model.predict(x_train)
pred_train = np.argmax(predictions, axis=1)
train_label = np.argmax(y_train, axis=1)
print('Training Accuracy', np.sum(pred_train == train_label) / N_train * 100)

# # Training Accuracy
# predictions = model.predict(x_valid)
# pred_valid = np.argmax(predictions, axis=1)
# valid_label = np.argmax(y_valid, axis=1)
# print('Validation Accuracy', np.sum(pred_valid == valid_label) / N_valid * 100)

# Test Accuracy
predictions = model.predict(x_test)
pred_test = np.argmax(predictions, axis=1)
test_label = np.argmax(y_test, axis=1)
print('Testing Accuracy', np.sum(pred_test == test_label) / N_test * 100)


# ### Save the model

# In[18]:

date = str(datetime.date.today() )
time = str(datetime.datetime.now().time())[:-7]

filename = '/home/shikhar/Datasets/Models/review_' + model_type + '_' + date + '_' +time;
with open( filename, 'wb') as output:
    pickle.dump([model.get_config(), model.get_weights(), model.history.history], output, pickle.HIGHEST_PROTOCOL)


# ### Plot the results

# In[15]:

import matplotlib.pyplot as plt
plt.style.use('ggplot')

plt.rcParams["figure.figsize"] = (10,6)
plt.plot(res.history['acc'])
plt.plot(res.history['loss'])
plt.plot(res.history['val_acc'])
plt.plot(res.history['val_loss'])
plt.legend(['Train acc','Train loss','Valid acc', 'Valid loss'], loc=2)
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.title('Using '+ model_type)
# imgName = 'Images/' + model_type + '_' + date + '_' + time + '.jpg'
# plt.savefig( imgName, dpi= 200, bbox_inches='tight', transparent=False)
plt.show()


# ### Continue from a pretrained model

# In[19]:

# Loading saved data
import pickle

# filename = '/home/shikhar/Datasets/Models/3_CNN-static_2017-04-04_11:38:41'
with open( filename, 'rb') as input:
    out = pickle.load(input)

model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(), metrics=["accuracy"])
model.set_weights(out[1])


# In[20]:

res = model.fit(x_train, y_train,
          batch_size = batch_size,
          epochs=300,
          validation_data=(x_test, y_test), verbose=2)


# ### Save the model (marking as continued)

# In[24]:

date = str(datetime.date.today() )
time = str(datetime.datetime.now().time())[:-7]

filename = '/home/shikhar/Datasets/Models/' + model_type + '_continued_' + date + '_' +time;
with open( filename, 'wb') as output:
    pickle.dump([model.get_config(), model.get_weights(), model.history.history], output, pickle.HIGHEST_PROTOCOL)


# In[23]:

import os
os.system('say done')


# In[24]:

files = [
    'CNN-rand_continued_2017-04-02_19:26:35',
    'CNN-rand_continued_2017-04-03_16:50:47',
    'CNN-rand_continued_2017-04-03_17:18:47'
]


# In[25]:

train_acc, train_loss, val_acc, val_loss = [],[],[],[]

for file in files:
    filename = '/home/shikhar/Datasets/Models/'  + file
    with open( filename, 'rb') as input:
        out = pickle.load(input)
    train_acc += out[2]['acc']
    train_loss += out[2]['loss']
    val_acc += out[2]['val_acc']
    val_loss += out[2]['val_loss']


# In[35]:

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = (10,6)
plt.plot(train_acc)
plt.plot(train_loss)
plt.plot(val_acc)
# plt.plot(val_loss)
plt.legend(['Train acc','Train loss','Valid acc', 'Valid loss'], loc=2)
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.title('Using '+ model_type)
date = str(datetime.date.today() )
time = str(datetime.datetime.now().time())[:-7]
imgName = 'Images/' + model_type + '_' + date + '_' + time + '.jpg'
plt.savefig( imgName, dpi= 200, bbox_inches='tight', transparent=False)
plt.show()



# coding: utf-8

# In[15]:

from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import numpy as np
import pickle, datetime


# In[2]:

phr_to_ind = dict()

with open('../Datasets/SST1_dataset/dictionary.txt') as f:
    for line in f:
        entry = line.split('|')
        phr_to_ind[entry[0]] = int(entry[1])

keys = phr_to_ind.keys();

print(len(phr_to_ind), phr_to_ind['Good'])


# In[3]:

# Without doing the below computation directly load the stored output
sentence_list = []
sentiment = []

with open('../Datasets/SST1_dataset/SentenceWithCorrection.txt') as f:
    for line in f:
        sent = line[:-1]
        sentence_list.append(sent)
        sentiment.append(phr_to_ind[sent])

print(len(sentence_list))


# In[4]:

ind_to_senti = dict()

with open('../Datasets/SST1_dataset/sentiment_labels.txt') as f:
    f.readline()
    for line in f:
        entry = line.split('|')
        ind_to_senti[int(entry[0])] = float(entry[1])

print(len(ind_to_senti))


# In[5]:

split_ind = []
with open('../Datasets/SST1_dataset/datasetSplit.txt') as f:
    f.readline()
    for line in f:
        entry = line.split(',')
        split_ind.append(int(entry[1]))

print(len(split_ind))

for i in range(len(split_ind)):
    if split_ind[i] == 3:
        split_ind[i] = 1
        
N_train = split_ind.count(1)
N_test = split_ind.count(2)
N_valid = split_ind.count(3)
print (N_train, N_test, N_valid)


# In[6]:

N_sent = len(sentence_list);
N_category = 5

y_label = []

for ind in sentiment:
    val = ind_to_senti[ind]
    if val >= 0.0 and val <= 0.2:
        y_label.append(0);
    elif val > 0.2 and val <= 0.4:
        y_label.append(1)
    elif val > 0.4 and val <= 0.6:
        y_label.append(2)
    elif val > 0.6 and val <= 0.8:
        y_label.append(3)
    else:
        y_label.append(4)

print(y_label.count(0), y_label.count(1), y_label.count(2), y_label.count(3))

# Labels in one-hot encoding
y_train = np.zeros((N_train, N_category), np.uint8)
y_test  = np.zeros((N_test , N_category), np.uint8)
y_valid = np.zeros((N_valid, N_category), np.uint8)

c1,c2,c3 = 0,0,0
for i in range(len(y_label)):
    label = y_label[i]
    if split_ind[i] == 1:
        y_train[c1, label] = 1;  c1 += 1
    elif split_ind[i] == 2:
        y_test [c2, label] = 1;  c2 += 1
    else:
        y_valid[c3, label] = 1;  c3 += 1


# In[7]:

x_all = []
max_sent_len = -1;
max_wrd_len = -1
wrd_to_ind = dict()

ind_new = 1;
for sent in sentence_list:
    wrds = sent.split()
    vec = []
    for wrd in wrds:
        if wrd not in wrd_to_ind.keys():
            wrd_to_ind[wrd] = ind_new
            ind_new += 1
            
        ind = wrd_to_ind[wrd]
        vec.append(ind)
            
    max_sent_len = max(len(vec), max_sent_len)
    x_all.append(vec)

# Get inverse dictionary
ind_to_wrd = dict((v, k) for k, v in wrd_to_ind.items())
ind_to_wrd[0] = "<PAD/>"

print(len(phr_to_ind), len(wrd_to_ind))


# In[8]:

x_train = np.zeros((N_train, max_sent_len), np.int32)
x_test  = np.zeros((N_test,  max_sent_len), np.int32)
x_valid = np.zeros((N_valid, max_sent_len), np.int32)

c1, c2, c3 = 0,0,0
for i in range(len(x_all)):
    vec = x_all[i]
    if split_ind[i] == 1:
        x_train[c1,0:len(vec)] = np.int32(vec); 
        c1 += 1
    elif split_ind[i] == 2:
        x_test [c2,0:len(vec)] = np.int32(vec); 
        c2 += 1
    else:
        x_valid[c3,0:len(vec)] = np.int32(vec); 
        c3 += 1

print(c1, c2, c3)


# In[9]:

batch_size = 32
maxlen = max_sent_len
max_features = 300


# In[10]:

# max_features = 20000
# maxlen = 80  # cut texts after this number of words (among top max_features most common words)
# batch_size = 32

# print('Loading data...')
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# print(len(x_train), 'train sequences')
# print(len(x_test), 'test sequences')

# print('Pad sequences (samples x time)')
# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
# print('x_train shape:', x_train.shape)
# print('x_test shape:', x_test.shape)


# In[11]:


# In[19]:

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(5, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
res = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=1,
          validation_data=(x_test, y_test), verbose=2)

score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)


# In[21]:


date = str(datetime.date.today() )
time = str(datetime.datetime.now().time())[:-7]

filename = './lstm_' + '_' + date + '_' +time;
with open( filename, 'wb') as output:
    pickle.dump([res.model.get_config(), res.model.get_weights(), res.history], output, pickle.HIGHEST_PROTOCOL)
    


# In[ ]:




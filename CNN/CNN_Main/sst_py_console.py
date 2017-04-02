import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Embedding, regularizers
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.merge import Concatenate
from keras import optimizers
from keras import backend as K
from w2v import train_word2vec 

import numpy as np
import difflib

phr_to_ind = dict()

with open('../../Datasets/SST1_dataset/dictionary.txt') as f:
    for line in f:
        entry = line.split('|')
        phr_to_ind[entry[0]] = int(entry[1])

keys = phr_to_ind.keys();

print(len(phr_to_ind), phr_to_ind['Good'])

# Without doing the below computation directly load the stored output
sentence_list = []
sentiment = []

with open('../../Datasets/SST1_dataset/SentenceWithCorrection.txt') as f:
    for line in f:
        sent = line[:-1]
        sentence_list.append(sent)
        sentiment.append(phr_to_ind[sent])

print(len(sentence_list))

# sentence_list = []
# sentiment = []

# with open('../../Datasets/SST1_dataset/datasetSentences.txt') as f:
#     f.readline()
#     for line in f:
#         entry = line.split('\t')
#         sent = entry[1][:-1]
#         sent = sent.replace('-LRB-', '(')
#         sent = sent.replace('-RRB-', ')')
    
#         if sent in phr_to_ind.keys():
#             sentiment.append(phr_to_ind[sent])
#         else:
#             print('.', end="")
#             keys_subset = [k for k in keys if (k[0] == sent[0])]
#             key = difflib.get_close_matches(sent, keys_subset, n=1);
#             sent = key[0]
#             sentiment.append(phr_to_ind[sent])
            
#         sentence_list.append(sent)
        
# print(len(sentence_list))

# # Written the output in a file
# f = open('../../Datasets/SST1_dataset/SentenceWithCorrection.txt', 'w')
# for sent in sentence_list:
#     f.write(sent + '\n')
# f.close()

ind_to_senti = dict()

with open('../../Datasets/SST1_dataset/sentiment_labels.txt') as f:
    f.readline()
    for line in f:
        entry = line.split('|')
        ind_to_senti[int(entry[0])] = float(entry[1])

print(len(ind_to_senti))

split_ind = []
with open('../../Datasets/SST1_dataset/datasetSplit.txt') as f:
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

# Training model
model_type    = 'CNN-static'  # CNN-rand|CNN-non-static|CNN-static
embedding_dim = 300         # word2vec dim
vocab_size    = len(ind_to_wrd)

if model_type in ['CNN-non-static', 'CNN-static']:
    embedding_wts = train_word2vec( np.vstack((x_train, x_test, x_valid)), 
                                    ind_to_wrd, num_features = embedding_dim)
    if model_type == 'CNN-static':
        x_train = embedding_wts[0][x_train]
        x_test  = embedding_wts[0][x_test]
        x_valid = embedding_wts[0][x_valid]
        
elif model_type == 'CNN-rand':
    embedding_wts = None
    
else:
    raise ValueError("Unknown model type")

batch_size   = 50
filter_sizes = [3,4,5]
num_filters  = 100
dropout_prob = (0.5, 0.8)
hidden_dims  = 100

l2_reg = 0.3
embedding_dim = 300

# Deciding dimension of input based on the model
input_shape = (max_sent_len, embedding_dim) if model_type == "CNN-static" else (max_sent_len,)
model_input = Input(shape = input_shape)

# Static model do not have embedding layer
if model_type == "CNN-static":
    z = Dropout(dropout_prob[0])(model_input)
else:
    z = Embedding(vocab_size, embedding_dim, input_length = max_sent_len, name="embedding")(model_input)
    z = Dropout(dropout_prob[0])(z)

# Convolution layers
z1 = Conv1D(    filters=100, kernel_size=3, 
                padding="valid", activation="relu", 
                strides=1)(z)
z1 = MaxPooling1D(pool_size=2)(z1)
z1 = Flatten()(z1)

z2 = Conv1D(    filters=100, kernel_size=4, 
                padding="valid", activation="relu", 
                strides=1)(z)
z2 = MaxPooling1D(pool_size=2)(z2)
z2 = Flatten()(z2)

z3 = Conv1D(    filters=100, kernel_size=5, 
                padding="valid", activation="relu",
                strides=1)(z)
z3 = MaxPooling1D(pool_size=2)(z3)
z3 = Flatten()(z3)

# Concatenate the output of all convolution layers
z = Concatenate()([z1, z2, z3])
z = Dropout(dropout_prob[1])(z)

# Dense(64, input_dim=64, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))

z = Dense(hidden_dims, activation="relu", kernel_regularizer=regularizers.l2(0.01))(z)
model_output = Dense(N_category, activation="sigmoid")(z)
    
model = Model(model_input, model_output)
model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adadelta(lr=1, decay=0.001), metrics=["accuracy"])
model.summary()

if model_type == "CNN-non-static":
    embedding_layer = model.get_layer("embedding")
    embedding_layer.set_weights(embedding_wts)

res = model.fit(x_train, y_train, 
          batch_size = batch_size,
          epochs=100,
          validation_data=(x_valid, y_valid), verbose=2)

# Training Accuracy
predictions = model.predict(x_train)
pred_train = np.argmax(predictions, axis=1)
train_label = np.argmax(y_train, axis=1)
print('Training Accuracy', np.sum(pred_train == train_label) / N_train * 100)

# Training Accuracy
predictions = model.predict(x_valid)
pred_valid = np.argmax(predictions, axis=1)
valid_label = np.argmax(y_valid, axis=1)
print('Validation Accuracy', np.sum(pred_valid == valid_label) / N_valid * 100)

# Test Accuracy
predictions = model.predict(x_test)
pred_test = np.argmax(predictions, axis=1)
test_label = np.argmax(y_test, axis=1)
print('Testing Accuracy', np.sum(pred_test == test_label) / N_test * 100)

import pickle, datetime

date = str(datetime.date.today() )
time = str(datetime.datetime.now().time())[:-7]

filename = './Datasets/Models/' + model_type + '_' + date + '_' +time;
with open( filename, 'wb') as output:
    pickle.dump([model.get_config(), model.get_weights(), model.history.history], output, pickle.HIGHEST_PROTOCOL)
    
## Loading saved data
# with open( filename, 'rb') as input:
#     out = pickle.load(input)
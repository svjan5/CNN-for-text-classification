
# coding: utf-8

# In[1]:




import os
import pdb
import numpy as np
import argparse
from collections import Counter
from sklearn import svm
import matplotlib.pyplot as plt
import scipy.sparse as sp


# # Data Loading and preprocessing

# ### Loading split information


split_ind = []
with open('../Datasets/SST1_dataset/datasetSplit.txt') as f:
    f.readline()
    for line in f:
        entry = line.split(',')
        split_ind.append(int(entry[1]))

print(len(split_ind))

# Merging validation set to training data
for i in range(len(split_ind)):
    if split_ind[i] == 3:
        split_ind[i] = 1
        
N_train = split_ind.count(1)
N_test = split_ind.count(2)
N_category = 5


# ### Phrase -> Index


phr_to_ind = dict()

with open('../Datasets/SST1_dataset/dictionary.txt') as f:
    for line in f:
        entry = line.split('|')
        phr_to_ind[entry[0]] = int(entry[1])

keys = phr_to_ind.keys();

print(len(phr_to_ind), phr_to_ind['Good'])


# ### Loading sentences


# Without doing the below computation directly load the stored output
x_train_sent = []
x_test_sent = []
sentiment = []

counter = 0
with open('../Datasets/SST1_dataset/SentenceWithCorrection.txt') as f:
    for line in f:
        sent = line[:-1]
        if(split_ind[counter] == 1):
            x_train_sent.append(sent)
        else:
            x_test_sent.append(sent)
        
        sentiment.append(phr_to_ind[sent])
        counter += 1

print(len(x_train_sent), len(x_test_sent))


# ### Loading sentiment information 


ind_to_senti = dict()

with open('../Datasets/SST1_dataset/sentiment_labels.txt') as f:
    f.readline()
    for line in f:
        entry = line.split('|')
        ind_to_senti[int(entry[0])] = float(entry[1])

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

y_train_org = np.zeros(0, dtype = int)
y_test_org = np.zeros(0, dtype = int)

for i in range(len(y_label)):
    label = y_label[i]
    if split_ind[i] == 1:
        y_train_org = np.append(y_train_org,label)
    else:
        y_test_org = np.append(y_test_org,label)
        
print(len(y_train_org), len(y_test_org))


# # Training model


# Tokenize operation
def tokenize(sentence, grams):
    words = sentence.split()
    tokens = []
    for gram in grams:
        for i in range(len(words) - gram + 1):
            tokens += ["_*_".join(words[i:i+gram])]
    return tokens


def compute_ratio(poscounts, negcounts, alpha=1):
    pos_keys = list(poscounts.keys())
    neg_keys = list(negcounts.keys())
    
    alltokens = list(set( pos_keys + neg_keys))
    dic = dict((t, i) for i, t in enumerate(alltokens))
    d = len(dic)
    p, q = np.ones(d) * alpha , np.ones(d) * alpha
    for t in alltokens:
        p[dic[t]] += poscounts[t]
        q[dic[t]] += negcounts[t]
    p /= abs(p).sum()
    q /= abs(q).sum()
    r = np.log(p/q)
    return dic, r


# ### Creating train and test input data


ngrams = [1,2,3]
max_token_num = -1;

# for sent in x_train_sent:
#     tokens = list(set(tokenize(sent, ngrams)))
#     max_token_num = max(max_token_num, len(tokens))


# for sent in x_test_sent:
#     tokens = list(set(tokenize(sent, ngrams)))
#     max_token_num = max(max_token_num, len(tokens))

# X_train = np.zeros((len(x_train_sent), max_token_num), np.float64)
# X_test = np.zeros((len(x_test_sent), max_token_num), np.float64)

# print(X_train.shape, X_test.shape)


# In[2]:

ngrams = [1,2,3]

svm_multi = svm.SVC(kernel = 'linear')


keywords = Counter()

counter = 0
for sent in x_train_sent:
    keywords.update(tokenize(sent, ngrams))
    counter += 1

alltokens = list(keywords.keys())
dic = dict((t, i) for i, t in enumerate(alltokens))
    
    
x_train = sp.lil_matrix((len(x_train_sent), len(dic)), dtype=np.float32)
    
counter = 0
for sent in x_train_sent:
    tokens = tokenize(sent, ngrams)
    indexes = []
    for t in tokens:
        try:
            indexes += [dic[t]]
        except KeyError:
            pass
    indexes = list(set(indexes))
    indexes.sort()

    for i in indexes:
        x_train[counter,i] = 1

    counter += 1
        
# Arrange test data
x_test = sp.lil_matrix((len(x_test_sent), len(dic)), dtype=np.float32)

counter = 0
for sent in x_test_sent:
    tokens = tokenize(sent, ngrams)
    indexes = []
    for t in tokens:
        try:
            indexes += [dic[t]]
        except KeyError:
            pass
    indexes = list(set(indexes))
    indexes.sort()

    data = []
    for i in indexes:
        x_test[counter, i] = 1

    counter = counter + 1

svm_multi.fit(x_train, y_train_org)

#supp_alpha = svm_oneVsAll[category].dual_coef_
#supp_vecs = svm_oneVsAll[category].support_vectors_
#supp_ind = svm_oneVsAll[category].support_
#b = svm_oneVsAll[category].intercept_
# w = np.zeros( (1,np.size(supp_vecs,1)), np.float32)
#w = sp.csr_matrix((1, np.size(supp_vecs,1)))
    
#    counter = 0
#    for i in supp_ind:
#        w = w + (supp_alpha[0,counter]*y_train[i])*supp_vecs[counter,:]
#        counter +=1
    
pred_train = svm_multi.predict(x_train)

#pred_train_prob = np.argmax(svm_multi.predict(x_train),axis = 1)
#pred_train1 = np.sign(w*x_train.T + b*np.ones(np.size(x_train,0)))
print( 'Train Accuracy', np.sum(pred_train == y_train_org)*1.0/ len(y_train_org))

#print( 'corr', np.sum(pred_train == pred_train_prob)*1.0/ len(y_train_org))

pred_test = svm_multi.predict(x_test)

#pred_test_prob = np.argmax(svm_multi.predict(x_test),axis = 1)
#pred_test1 = np.sign(w*x_test.T + b*np.ones(np.size(x_test,0)))
print( 'Test Accuracy', np.sum(pred_test == y_test_org)*1.0/ len(y_test_org))
#print( 'corr', np.sum(pred_test == pred_test_prob)*1.0/ len(y_train_org))


# In[46]:

count = np.zeros((5,), dtype=int)
for i in range(len(y_test_org)):
    if y_test_org[i] == pred_test[i]:
            count[y_test_org[i]] +=1

print('Class wise accuracies: ')
for i in range(5):
    print(count[i]*1.0/np.sum(y_test_org == i))
    
print('------------------------------------------------')

conf_mat = np.zeros((5,5), dtype = np.float32)
for i in range(len(y_test_org)):
    conf_mat[y_test_org[i],pred_test[i]] += 1.0/np.sum(y_test_org == y_test_org[i])

plt.pcolor(conf_mat)
plt.xticks(np.arange(5)+0.5, ['very negative', 'negative' , 'neutral' , 'positive' , 'very positive'])
plt.yticks(np.arange(5)+0.5, ['very negative', 'negative' , 'neutral' , 'positive' , 'very positive'])



#pred_maj = np.sum(pred_oneVsAll, axis=1)
#print(pred_maj.shape)


# ## Computing $w^{T}x +b$ 


#category = 0
#
#for i in range(len(y_train)):
#    y_train[i] = (-1) if y_train_org[i] == category else (1)
#    
#supp_alpha = svm_oneVsAll[0].dual_coef_
#supp_vecs = svm_oneVsAll[0].support_vectors_
#supp_ind = svm_oneVsAll[0].support_
#b = svm_oneVsAll[0].intercept_
## w = np.zeros( (1,np.size(supp_vecs,1)), np.float32)
#w = sp.csr_matrix((1, np.size(supp_vecs,1)))
#
#counter = 0
#for i in supp_ind:
#    w = w + (supp_alpha[0,counter]*y_train[i])*supp_vecs[counter,:]
#    counter +=1


# In[ ]:




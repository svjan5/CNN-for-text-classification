# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 12:20:59 2017

@author: Santhosh
"""
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt

#the story and structure are well honed 

data = [('it', 0.094893411, 0.067789622, 0.24174465),
 ("'s", 0.16971132, 0.1769509, 0.250965),
 ('fun', 0.26758641, 0.29317465, 0.25750187),
 ('lite', 0.30395886, 0.27859962, 0.16361547)]
 

word = []
val = np.zeros((3,len(data)), dtype = np.float32)
for k in range(len(data)):
    word.append(data[k][0])
    val[0][k] = float(data[k][1])
    val[1][k] = float(data[k][2])
    val[2][k] = float(data[k][3])

val = np.abs(val)
pos = list(range(len(data)))
width = 0.25
plt.style.use('ggplot')

tableau20 = [(216, 37, 38), (242, 108, 100), (174, 199, 232), (152, 223, 138),    
             (44, 160, 44)]
             
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)

plt.bar(pos,
        #using df['pre_score'] data,
        val[0][:],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color=(1,0,0),
        # with label the first value in first_name
        label='Back Propagation')

# Create a bar with mid_score data,
# in position pos + some width buffer,
plt.bar([p + width for p in pos],
        #using df['mid_score'] data,
        val[1][:],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color=(0,1,0),
        # with label the second value in first_name
        label='Guided Back propagation')

# Create a bar with post_score data,
# in position pos + some width buffer,
plt.bar([p + width*2 for p in pos],
        #using df['post_score'] data,
        val[2][:],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color=(0,0,1),
        # with label the third value in first_name
        label='Deconvnet')



plt.xticks(np.arange(len(data))+0.25+0.125, word)
plt.legend(['BackProp','Guided-BackProp','Deconvnet'])
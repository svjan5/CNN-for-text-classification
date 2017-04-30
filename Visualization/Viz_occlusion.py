# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 12:01:57 2017

@author: Santhosh
"""

import numpy as np
import matplotlib.pyplot as plt

tableau20 = [(216, 37, 38), (242, 108, 100), (174, 199, 232), (152, 223, 138),    
             (44, 160, 44)]
             
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)

hist = np.array([0.08847391,  0.20979726,  0.11424229,  0.27708733 , 0.3103992])
plt.style.use('ggplot')
barplots = plt.bar(range(len(hist)),hist)
for i in range(5):
    barplots[i].set_color(tableau20[i])
plt.xticks(np.arange(5)+0.5, ['very negative', 'negative' , 'neutral' , 'positive' , 'very positive'])
plt.savefig( 'Viz_1_4000_org', dpi= 200, bbox_inches='tight', transparent=False)
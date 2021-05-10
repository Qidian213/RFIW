# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 21:20:28 2020

@author: USER
"""
import random
import pandas as pd
import numpy as np

if __name__ == '__main__':
    csv_file_lists = ["results/val_p1_f10025_300_aug_db_re50_0.7888_037.csv"]
                      
    indexs = []
    scores = []
    preds  = []
    
    weight = 1.0/len(csv_file_lists)
    for csv_file in csv_file_lists:
        file = pd.read_csv(csv_file)
        df   = pd.DataFrame(file)

        for i in range(len(df)):
            if(len(indexs) < len(df)):
                indexs.append(df['index'][i])
                scores.append(weight*df['label'][i])
            else:
                scores[i] += weight*df['label'][i]

    for score in scores:
        if(score>=0.5):
            preds.append(1)
        else:
            preds.append(0)

    dataframe = pd.DataFrame({'index':indexs,'label':scores, 'pred': preds})
    dataframe.to_csv("Final.csv",index=False,sep=',')

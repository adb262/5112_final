#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 15:20:40 2020

@author: rashmisinha
"""

import numpy as np
import pandas as pd
from sitq import Mips
from sklearn.preprocessing import MinMaxScaler
import time
import random

train = pd.read_csv("SpotifyFeatures.csv")
train_df = train

col_names = ['acousticness',
               'danceability',
               'energy',
               'instrumentalness',
               'liveness',
               'loudness',
               'speechiness',
               'tempo',
               'valence',
               'popularity']
train = train.drop_duplicates(col_names, keep='first')
train = train[col_names]

print(train.shape[0], train.shape[1])


scaler = MinMaxScaler()
scaled = pd.DataFrame(scaler.fit_transform(train))
train_arr = scaled.to_numpy()

randomindex = random.sample(range(1, 100), 3)

'''
my playlist

shape of you - 108001 - pop
burn - 110083 - pop
despacito - 109448 - pop/dance
elastic heart - 13673 - dance/pop
wide awake - 111809 - pop/dance
Lean On - 13982 - Rap/pop
'''

mips = Mips(signature_size=4)

# Learn lookup table and parameters for search
mips.fit(train_arr)

selected_song_index = 110961
query_arr = train_arr[selected_song_index]

# Find items which are likely to maximize inner product against query
start = time.time()
item_indexes, scores = mips.search(query_arr, limit=5, distance=1)
end = time.time()

trainnp = train_df.to_numpy()
selected_song_info = [trainnp[selected_song_index][2], trainnp[selected_song_index][0]]
selected_song = trainnp[selected_song_index]
query_song_vector = [selected_song[4], selected_song[5], selected_song[6],
                        selected_song[8], selected_song[9], selected_song[11], 
                        selected_song[12], selected_song[14], selected_song[15],
                        selected_song[17]]

song_vector = [0,0,0,0,0,0,0,0,0,0]
songs=[]

for index in item_indexes:
    song = trainnp[index]
    songs_data = train_df[(train_df['artist_name'] == song[1]) & (train_df['track_name'] == song[2])]
    song_info = [song[1],song[2]]
    recommended_song_vector = [song[4], song[5], song[6],
                        song[8], song[9], song[11], 
                        song[12], song[14], song[15],
                        song[17]]
    song_vector = [song_vector[i] + recommended_song_vector[i] for i in range(len(song_vector))]
    for index, row in songs_data.iterrows():
      song_info.append(row['genre']) 
    songs.append(song_info)

song_vector = [song_vector[i]/5 for i in range(len(song_vector))]
print(selected_song_info)
print(songs)

print("Latency (ms):", 1000*(end - start))
print(query_song_vector/np.linalg.norm(query_song_vector))
print(song_vector/np.linalg.norm(song_vector))

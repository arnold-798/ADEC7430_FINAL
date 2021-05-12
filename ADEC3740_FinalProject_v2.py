#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 10:25:47 2021

@author: chrisarnold
"""

#%% Big Data Econometrics Final Project

# Import General Packages

import sys
import os

# Import Packages for the Project

import streamlit as st
import pandas as pd
import numpy as np
import re
import datetime as dt
import inspect
import pickle

import pkg_resources
import gensim
from gensim import models # https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4
import gensim.downloader as api
import tqdm


#%% Title for the Webpage / Dashboard / UI

st.title("""
         
         Big Data Econometrics Spring 2021 Project         
                
         ***
         Our Webpage Allows You to Input Various Sentence Queries and Returns the Closest Answer!
         ***
                 
         """
         )

#%% Instructions Page


def instructions():
    st.header("Instructions for our very Impressive Project")
    st.subheader("Input some text in the form of an equation")
    


#%% Load in word embeddings data frame

## Ingest the two google news word embeddings dataset - Should only be run during development; the created index dataset is what should be
## Saveed to the github repository

#@st.cache
#def load_embeddings(): 
#    wv = api.load(name = 'word2vec-google-news-300') # Google News Dataset ~ 100 billion words
#    return wv

#wv = api.load(name = 'glove-wiki-gigaword-50')

# Problem: File cannot be uploaded to the github repository - is there another way we can store it online rather than download it
# each time someone opens the website?

#cwd = os.getcwd()

#new_path = os.path.join(cwd, '/temp_folder/word2vec-google-news-300.pkl')

#os.mkdir(new_path)

## this should only be run once, after downloading the big data above
#t0 = dt.datetime.now()
#with open("/Users/chrisarnold/Desktop/Big_Data_Econometrics/PyEnvs/final_project/glove-wiki-gigaword-50.pkl", 'wb') as tf:
#  pickle.dump(wv, tf)
#t1 = dt.datetime.now()
#print("This took: ", (t1-t0))
#st.write(("Loading data took: " + (t1-t0)))

#t0 = dt.datetime.now()

github_pkl = "https://github.com/arnold-798/ADEC7430_FINAL/raw/main/glove-wiki-gigaword-50.pkl"

wv = pd.read_pickle(github_pkl, compression='infer', storage_options=None)

#with open("/Users/chrisarnold/Desktop/Big_Data_Econometrics/PyEnvs/final_project/glove-wiki-gigaword-50.pkl", 'rb') as tf:
#  wv = pickle.load(tf)

#t1 = dt.datetime.now()
#print("This took: ", (t1-t0))
#st.write('Loading data took: ',t1-t0)

#%% Inspect the gensim package laoded

word2vec = wv # could choose wv2 for comparison, later
#dir(word2vec)

## This tells us that to recover the embedding of a word, we need to:
    # 1. get the index of the word in the vocab
    # 2. get the row in word2vec.vectors with that index 

#print(inspect.getsource(word2vec.get_vector))

wv_dict = word2vec.key_to_index.items()

# We want a dataframe with word, index
tdict = dict(word = [k for k,v in word2vec.key_to_index.items()],
             index = [v for k,v in word2vec.key_to_index.items()])

# this is the dataframe of word vs index
#@@ save this and load it when implementing the "search engine"
word_index_df = pd.DataFrame(tdict)

#%% Implement a function to run the code with

# NOw: implement the function which, given a string like sample1-3 above (top)
# does:
# 1. looks up for each word the embedding vector (hint: use word_index_dt)
  # to ponder: lowercase or not the word??? see the test below...
# 2. multiplies by the weights vector 
# 3. applies signs
# 4. adds the vectors together to get a single 300-dim vector
#.....

sample = "5*king - man + 2*woman"
#sample1 = "+king-man+woman" 
#sample2 = "+5*king - 1*man + 2*woman"
#sample3 = "+5.3*king - 1.2*man + 2.2*woman" 

#%% Create the parse function

# Parse function ingests the input query and spits out a pandas DataFrame that allows us to analyze the string

def parse(inpt, DEBUG=False):
  inpt1 = re.sub(r"\s","", inpt) # get rid of white spaces
  if inpt1[0] not in ["+", "-"]: # add sign to start of string
    inpt1 = "+" + inpt1
  inpt1 = re.sub("(?P<sign>[+-])(?P<let>[a-zA-Z])","\g<sign>1*\g<let>", inpt1)
  if DEBUG:
    print(inpt1)
  wordlist = re.findall("[A-Za-z]+", inpt1) # returns all words in string
  signlist = re.findall("[+-]", inpt1) # returns all signs in string
  numlist = re.findall("[\d.]+", inpt1) # returns all numbers in string
  numlist = [float(i) for i in numlist] # convert string numbers to floats
  if DEBUG:
      print(wordlist,signlist, numlist)
  for i in range(0,len(numlist)):
    if signlist[i] == '-':
      numlist[i] = (-1)*numlist[i]
  #pstring_dict = {'Original_String':inpt1, 'Words_List':wordlist, 'Signs_List':signlist, 'Coef_List':numlist} 
  if DEBUG:
      print(numlist)
  pstring_dictv2 = {'Words_List':wordlist, 'Coef_List':numlist} 
  # Chris: I changed the output of the parse function to a dictionary so we could reference the ouptut more readily 
  # in the "create_vector" function below
  #pstring_df = pd.DataFrame(pstring_dict)
  return pstring_dictv2

#%%
#parse()
#%% Test out the parse function

#print(parse(sample))
#print(parse(sample1))
#print(parse(sample2))
#print(parse(sample3))

#sample_test = parse(sample)
#sample_words = sample_test['Words_List']

# Parse the input string and grab the individual words, signs and numbers     
#parsed_string = parse(sample)
#word_list = list(parsed_string['Words_List'])
#sign_list = list(parsed_string['Signs_List'])
#num_list = list(parsed_string['Coef_List'])


#%% Build out function 

#test_word = 'king'
#num_list_test = [2,3]

#word_index = word_index_df[word_index_df['word'] == test_word]
#word_vector = word2vec.vectors[word_index['index']]

#adj_word_vec = (map(lambda x: word_vector * x, num_list_test))

#adj_word_vec_test = word_vector * -2

#unique_wordvec = np.array(sum(adj_word_vec))

    
#%% Test code for the create vector function

parsed_string = parse(sample)

# Parse the input string and grab the individual words, signs and numbers     
word_list = list(parsed_string['Words_List'])
num_list = list(parsed_string['Coef_List'])

# Loop through each word in the list to find the associated index in the word embedings dataframe

for i in range(len(word_list)):
        
    indiv_word = word_list[i]
    
    word_index = word_index_df[word_index_df['word'] == indiv_word]
    word_vector = word2vec.vectors[word_index['index']]
    
    adj_word_vec = word_vector * num_list[i]
    
    #adj_word_vec = map(lambda x: word_vector * x, num_list)
    
    if i == 0:
        unique_wordvec = adj_word_vec
    else:
        unique_wordvec += adj_word_vec

    
#%% Create the get vector function 

# create_vector function takes the parsed input string, applys the coefficients, 

def create_vector(parsed_string, DEBUG=False):
    
    # Parse the input string and grab the individual words, signs and numbers     
    word_list = parsed_string['Words_List']
    num_list = parsed_string['Coef_List']
    if DEBUG:
        print(word_list)
        print(num_list)
        print(parsed_string)
    # Loop through each word in the list to find the associated index in the word embedings dataframe
        print(len(word_list))
    for i in range(len(word_list)):
            
        indiv_word = word_list[i]
        
        word_index = word_index_df[word_index_df['word'] == indiv_word]
        word_vector = word2vec.vectors[word_index['index']]
        adj_word_vec = word_vector * num_list[i]
        if DEBUG:
            print('indiv_word:',indiv_word) 
            print('word_vector', word_vector)
            print('adj_word_vec', adj_word_vec)
        
        #adj_word_vec = map(lambda x: word_vector * x, num_list)
        
        if i == 0:
            unique_wordvec = adj_word_vec
        else:
            unique_wordvec += adj_word_vec
            
    unique_wordvec_v1 = unique_wordvec[0]
            
    return unique_wordvec_v1
    
#%%
# v1 = create_vector(parse("king-man+5*woman"),DEBUG=False)
# v2 = create_vector(parse("-man+king+5*woman"),DEBUG=False)
# print(np.max(np.abs(v1-v2)))
# print(v1, v2)
# parse("king-man+5*woman",DEBUG=True)
# parse("-man+king+5*woman",DEBUG=True)
        
#%% Test out create_vector function on sample string - These functions will build the larger one

test = parse(sample)

unique_wordvec = create_vector(test)

#%% Problem #1 to discuss:    

## Problem: Should we consider uppercase vs lowercase 
#word2vec.vocab['King']
#word_index_df.loc[word_index_df.word.isin(['king','King','KING'])]

#%% Compute the lengths of the word embeddings data in blocks

# normalize the embeddings
# create a new numpy array storing the square root of the sum of squares ("length") for each row in "vectors"
norms = np.zeros(word2vec.vectors.shape[0])
vectors_norm = word2vec.vectors.copy()

# check/example for division of numpy arrays with broadcasting
#x1 = np.array([[1,2],[3,4],[5,6]])
#x2 = np.array([5,6,7])
#print(x1)
#print(x2)
#np.divide(x1,x2.reshape(-1,1))
#np.floor_divide(x1, x2) # for integer division

# compute lengths in blocks
t0 = dt.datetime.now()
blocksize = 100
blocks = int(np.ceil(word2vec.vectors.shape[0]/blocksize))
for i in range(blocks):
# for i in range(100):
  # print(i)
  # i = 0
  # select the block
  start_index = i*blocksize
  end_index = min((i+1)*blocksize, word2vec.vectors.shape[0]) # stop at the end of the vector to avoid out of range errors
  tblock = word2vec.vectors[start_index:end_index]
  xx = np.sqrt(np.diag(np.dot(tblock, tblock.T)))
  #xx.shape
  norms[start_index:end_index] = xx
  vectors_norm[start_index:end_index] = np.divide(vectors_norm[start_index:end_index], xx.reshape(-1,1))
t1 = dt.datetime.now()
print('this took ',t1-t0)

# test the normalized vectors
#np.dot(vectors_norm[127], vectors_norm[127])

#@@ save vectors_norm, load them at runtime for "search engine"

# clean-up / save RAM
del wv, norms

#%% Test the code that builds the match_vector function below

# THEN: given a vector, parse the embeddings to find the best/closest matching index
# with that index, go and get the word from word_index_df
#fakevector = np.arange(300)
#fakevector = word2vec.get_vector('king')
#fakevector_norm = fakevector/(np.dot(fakevector, fakevector))
# how to parse 3000000 candidates?
# use the dot product
# the following naive way doesn't seem to work - not enough RAM
# xx = np.dot(vectors_norm, fakevector_norm)
#blocksize = 10000
#blocks = int(np.ceil(vectors_norm.shape[0]/blocksize))
#tdict = {} # dictionary to keep track of each block's candidate for closest vector
#for i in range(blocks):
#  # i = 0
#  start_index = i*blocksize
#  end_index = min((i+1)*blocksize, vectors_norm.shape[0])
#  # select block
#  x1 = np.dot(vectors_norm[start_index:end_index], fakevector_norm)
#  pos1 = np.argmax(x1)
#  tdict[i] = (pos1 + i*blocksize, x1[pos1]) # think why adding i*blocksize
#  # tdict
  
#tdict
#tlist = list(tdict.values())
#dist_blocks = pd.DataFrame(dict(pos = [k for (k,v) in tlist], dist=[v for (k,v) in tlist]))
#maxpos = np.argmax(dist_blocks.dist)
#real_max_pos = int(dist_blocks.iloc[maxpos]['pos'])

#top_answer = word_index_df[word_index_df.index == real_max_pos]

#%% 

#top_five_pos = dist_blocks.nlargest(5, 'dist')

#top_five_pos = top_five_pos.rename(columns={'pos':'index'})

#top_five_pos_index = top_five_pos[top_five_pos['pos']

#top_five_pos_list = list(top_five_pos_index)

#inner_join = pd.merge(word_index_df, top_five_pos, on = ['index']) 

#%% Function to take output from create_vector function, find closest matching embedding, related index, and then associated word


def match_vector(unique_wordvec, response_nmb = 5):
    
    # Check if the vector is normalized; if not, normalize the input vector
    normalized_test = np.sqrt(np.dot(unique_wordvec, unique_wordvec))
    
    if np.abs(1 - normalized_test) > 0.01:
        unique_vecnorm = unique_wordvec/ normalized_test
    else:
        unique_vecnorm = unique_wordvec
        
    
    blocksize = 10000
    blocks = int(np.ceil(vectors_norm.shape[0]/blocksize))
    top_block_list = []
    

    
    for i in range(blocks):
        # i = 0
        start_index = i*blocksize
        end_index = min((i+1)*blocksize, vectors_norm.shape[0])
        # select block
        x1 = np.dot(vectors_norm[start_index:end_index], unique_vecnorm)
        
        x2 = pd.DataFrame(dict(idx=np.arange(start_index, end_index), sim=x1))
        
        x3 = x2.sort_values(by='sim', ascending=False)
        
        top_five_pos = x3.nlargest(response_nmb, 'sim')

        top_block_list.append(top_five_pos)
    
    top_of_blocks = pd.concat(top_block_list)
    
    top_of_top = top_of_blocks.nlargest(response_nmb, 'sim')
    
    top_of_top.rename(columns={'idx':'index'}, inplace=True)

    inner_join = pd.merge(word_index_df, top_of_top, on = ['index'])
    
    inner_join = inner_join.sort_values(by='sim', ascending=False)
    
    # Should return a dictionary to be referenced 
       
    return inner_join

#%% Test the create vector function



#%% Final function that pulls previously constructed functions together, ingests input string and returns top five answers

# Choose top answer or top 5 with distances option

def query_answer(input_string, response_nmb = 5):
    
    parsed_string = parse(input_string)
    unique_vec = create_vector(parsed_string)
    response = match_vector(unique_vec, response_nmb)
    
    return response
            
#%% 
    
def use_tool():
    st.header("Test Out Gensim Word Embeddings")
    
    text_query_str = st.text_input(label = 'Insert your word query here:', help = "Example: 2.2*King - 3.1*Man + 5.7*Woman")
        
    st.text("Both response types display the distances")
    
    response_nmb = int(st.text_input('Select how many answers to see:', value = '3', help = "Insert a integer for how many results"))

    start_button = st.button("Run", key = "Run_Query_Answer")
    
    if start_button:
   
        # Insert Jeopardy waiting music here: 
        #st.audio(data, start_time = t0)
        
        response = query_answer(text_query_str, response_nmb=response_nmb)
        st.success("Done!")
        
        st.write(response_nmb, ":", response)
        
#%% Create the app with loaded data


def main():
    selected_box = st.sidebar.selectbox('Choose one of the following', 
                                        ('Tutotrial and Instructions', 'Use the Tool!'))
    if selected_box == 'Tutotrial and Instructions': 
        instructions() 
    if selected_box == 'Use the Tool!':
        use_tool()

    
if __name__ == "__main__":
    main()






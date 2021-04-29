#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 22:46:13 2021

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


#%% Title for the Webpage / Dashboard / UI

st.title("""
         
         Big Data Econometrics Spring 2021 Project         
                
         ***
         Our Webpage Allows You to Input Various Sentence Queries and Returns the Closest Answer!
         ***
         
        
         """
         )

#%% Textbook input Query

st.header("""
          
          Start having fun!
          
          ***Example: “king – man + woman =… (queen)”***
          
          """
          )


input_text = st.text_input("Input You're Text Query Here:")

#%% Ingest the input text into the function 

def word_ingest(input_text):
    text = str(input_text)
    return st.text(text)







import streamlit as st

import numpy as np
import pandas as pd

import requests
from bs4 import BeautifulSoup

import re
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


'''
# BERTerReads

---
'''


@st.cache(allow_output_mutation=True)
def load_model():
    '''
    Function to load (and cache) DistilBERT model 
    '''
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    return model


@st.cache(allow_output_mutation=True)
def get_reviews(url):
    '''
    Function to scrape all the reviews from the first page of a GoodReads book URL
    '''

    # Download & soupify webpage
    r = requests.get(url)
    soup = BeautifulSoup(r.content, features='html.parser')

    # Find all review text blocks
    reviews_src = soup.find_all('div', class_='reviewText stacked')

    # Initialize list to store cleaned review text
    reviews = []

    # Loop through each review text block
    for review in reviews_src:

        # Extract review text
        try:
            text = review.find('span', style='display:none').get_text(' ', strip=True)
        except:
            text = review.get_text(' ', strip=True)

        # Remove spoiler tags from review text
        text = re.sub(r'\(view spoiler\) \[', '', text)
        text = re.sub(r'\(hide spoiler\) \] ', '', text)

        # Append review text to list
        reviews.append(text)

    # Transform review list to dataframe
    df = pd.DataFrame(reviews, columns=['review'])
    
    return df


@st.cache
def clean_reviews(df):
    '''
    Function to clean review text and divide into individual sentences
    '''

    # Append space to all sentence end characters
    df['review'] = df['review'].str.replace('.', '. ').replace('!', '! ').replace('?', '? ')

    # Initialize dataframe to store review sentences
    sentences_df = pd.DataFrame()

    # Loop through each review
    for i in range(len(df)):

        # Save review to variable
        review = df.iloc[i]['review']

        # Tokenize review into sentences
        sentences = sent_tokenize(review)

        # Transform sentences into dataframe
        new_sentences = pd.DataFrame(sentences, columns=['sentence'])

        # Add sentences to sentences dataframe
        sentences_df = sentences_df.append(new_sentences, ignore_index=True)

    # Set lower and upper thresholds for sentence word count
    lower_thresh = 5
    upper_thresh = 50

    # Remove whitespaces at the start and end of sentences
    sentences_df['sentence'] = sentences_df['sentence'].str.strip()

    # Create list of sentence lengths
    sentence_lengths = sentences_df['sentence'].str.split(' ').map(len)

    # Filter sentences
    sentences_df = sentences_df[
        (sentence_lengths > lower_thresh) & (sentence_lengths < upper_thresh)]

    sentences_df.reset_index(drop=True, inplace=True)
    
    return sentences_df['sentence']


@st.cache
def embed_sentences(sentences):
    '''
    Function to transform sentences into vectors
    '''
    sentence_vectors = model.encode(sentences)

    return sentence_vectors


@st.cache
def get_opinions(sentences, sentence_vectors, k=3, n=1):
    '''
    Function to extract the n most representative sentences from k clusters, with density scores
    '''
    
    # Instantiate the model
    kmeans_model = KMeans(n_clusters=k, random_state=24)

    # Fit the model
    kmeans_model.fit(sentence_vectors);
    
    # Set the number of cluster centre points to look at when calculating density score
    centre_points = int(len(sentences) * 0.02)
    
    # Initialize list to store mean inner product value for each cluster
    cluster_density_scores = []
    
    # Initialize dataframe to store cluster centre sentences
    df = pd.DataFrame()

    # Loop through number of clusters
    for i in range(k):

        # Define cluster centre
        centre = kmeans_model.cluster_centers_[i]

        # Calculate inner product of cluster centre and sentence vectors
        ips = np.inner(centre, sentence_vectors)

        # Find the sentences with the highest inner products
        top_index = pd.Series(ips).nlargest(n).index
        top_sentence = sentences[top_index].iloc[0]
        
        centre_ips = pd.Series(ips).nlargest(centre_points)
        density_score = round(np.mean(centre_ips), 5)
        
        # Create new row with cluster's top 10 sentences and density score
        new_row = pd.Series([top_sentence, density_score])
        
        # Append new row to master dataframe
        df = df.append(new_row, ignore_index=True)

    # Rename dataframe columns
    df.columns = ['sentence', 'density']

    # Sort dataframe by density score, from highest to lowest
    df = df.sort_values(by='density', ascending=False).reset_index(drop=True)
    
    return df


# Load DistilBERT model
model = load_model()

# Retrieve URL from user
url = st.text_input('Input GoodReads book URL:')

# If URL has been entered...
if url != '':

    # Scrape reviews from URL
    reviews_df = get_reviews(url)

    # Divide reviews into their individual sentences
    sentences = clean_reviews(reviews_df)

    # Transform sentences into vectors
    sentence_vectors = embed_sentences(sentences)

    # Find the 3 most representative sentences
    df = get_opinions(sentences, sentence_vectors, 3)

    # Print out these sentences
    for i in range(len(df)):
        f"**Opinion #{i+1}:** {df['sentence'][i]}\n"
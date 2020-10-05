import streamlit as st

import numpy as np
import pandas as pd

import requests
from bs4 import BeautifulSoup

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

    r = requests.get(url)
    soup = BeautifulSoup(r.content, features='html.parser')

    reviews_src = soup.find_all('div', class_='reviewText stacked')

    reviews = []

    for review in reviews_src:

        reviews.append(review.text)

    df = pd.DataFrame(reviews, columns=['review'])
    
    return df


@st.cache
def clean_reviews(df):
    '''
    Function to clean review text and divide into individual sentences
    '''

    # Define spoiler marker & "...more" strings, and remove from all reviews
    spoiler_str_gr = '                    This review has been hidden because it contains spoilers. To view it,\n                    click here.\n\n\n'
    more_str = '\n...more\n\n'
    df['review'] = df['review'].str.replace(spoiler_str_gr, '')
    df['review'] = df['review'].str.replace(more_str, '')

    # Scraped reviews from GoodReads typically repeat the first ~500 characters
    # The following loop removes these repeated characters

    # Loop through each row in dataframe
    for i in range(len(df)):

        # Save review and review's first ~250 characters to variables
        review = df.iloc[i]['review']
        review_start = review[2:250]

        # Loop through all of review's subsequent character strings
        for j in range(3, len(review)):

            # Check if string starts with same sequence as review start
            if review[j:].startswith(review_start):
                # If so, chop off all previous characters from review
                df.at[i, 'review'] = review[j:]

    # Replace all new line characters
    df['review'] = df['review'].str.replace('\n', ' ')

    # Append space to all sentence end characters
    df['review'] = df['review'].str.replace('.', '. ').replace('!', '! ').replace('?', '? ')

    # Initialize dataframe to store review sentences, and counter
    sentences_df = pd.DataFrame()

    # Loop through each review
    for i in range(len(df)):

        # Save row and review to variables
        row = df.iloc[i]
        review = row.loc['review']

        # Tokenize review into sentences
        sentences = sent_tokenize(review)

        # Loop through each sentence in list of tokenized sentences
        for sentence in sentences:
            # Add row for sentence to sentences dataframe
            new_row = row.copy()
            new_row.at['review'] = sentence
            sentences_df = sentences_df.append(new_row, ignore_index=True)

    sentences_df.rename(columns={'review':'sentence'}, inplace=True)

    lower_thresh = 5
    upper_thresh = 50

    # Remove whitespaces at the start and end of sentences
    sentences_df['sentence'] = sentences_df['sentence'].str.strip()

    # Create list of sentence lengths
    sentence_lengths = sentences_df['sentence'].str.split(' ').map(len)

    num_short = (sentence_lengths <= lower_thresh).sum()
    num_long = (sentence_lengths >= upper_thresh).sum()
    num_sents = num_short + num_long

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
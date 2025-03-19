

import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt

import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
from gensim.models.coherencemodel import CoherenceModel



def plot_top_topics(df, n=10):
    #df_manifesto["topic"] = topics

    # Group by party and topic; count the number of times each topic was mentioned
    #party_topic_counts = df_manifesto.groupby(["party", "topic"]).size().unstack().fillna(0)
    #party_topic_counts = party_topic_counts.div(df_manifesto["party_count"], axis=0)
    
    # Get the top 10 topics
    top_n_topics = df.sum().nlargest(n + 1).index[1:]

    # Filter the DataFrame to include only the top 10 topics
    party_topic_counts_top_n = df[top_n_topics]

    # Plot the filtered DataFrame
    party_topic_counts_top_n.plot(kind="bar", stacked=True, figsize=(12, 6))
    plt.show()
    return party_topic_counts_top_n

def plot_topic_n_counts(df, topic_number):
    # Filter the DataFrame to include only the specified topic
    topic_n_counts = df[[topic_number]]
    
    # Plot the results
    topic_n_counts.plot(kind="bar", stacked=True, figsize=(12, 6))
    plt.show()

def limit_to_cmp(df, cmp_min, cmp_max):
    #new_df = new_df.dropna(subset=['cmp_code'])
    new_df = df[(df['cmp_code'] >= cmp_min) & (df['cmp_code'] <= cmp_max)]
    return new_df


def get_topics(df, codes):

    # Initialize an empty dictionary for topic_filter
    new_dict = {}

    # Filter out rows with NaN values in the 'domain_name' column
    filtered_codes = codes.dropna(subset=['title'])

    # Iterate through each distinct value in filtered_codes['domain_name']
    for domain in filtered_codes['title'].unique():
        # Get the cmp_codes for the current domain
        domain_codes = filtered_codes[filtered_codes['title'] == domain]['code']
        
        # Filter df_manifesto to get the topics for the current domain
        domain_topics = df[df['cmp_code'].isin(domain_codes)]['topic'].tolist()
        # Remove duplicate topics
        domain_topics = list(set(domain_topics))
        # Add the topics as a new column in topic_filter
        new_dict[domain] = [domain_topics][0]

    return new_dict

def split_text_by_date(df, split_date):
    new_rows = []
    for index, row in df.iterrows():
        if row['date'] < split_date:
            sentences = nltk.sent_tokenize(row['text'])
            for i, sentence in enumerate(sentences):
                # Check if the sentence ends with a digit followed by a period
                if re.search(r'\d+\.$', sentence) and i + 1 < len(sentences) and sentences[i + 1][0].isupper():
                    sentence += ' ' + sentences[i + 1]
                    sentences[i + 1] = sentence
                    continue
                # Disregard abbreviations like "bzw.", "z.B.", "u.a."
                if re.search(r'\b(bzw|z\.B|u\.a)\.$', sentence):
                    new_row = row.copy()
                    new_row['text'] = sentence
                    new_rows.append(new_row)
                    continue
                new_row = row.copy()
                new_row['text'] = sentence
                new_rows.append(new_row)
        else:
            new_rows.append(row)
    # Filter out documents that do not have a space character and documents that are shorter than 6 characters
    new_rows = [row for row in new_rows if ' ' in row['text'] and len(row['text']) >= 6]
    return pd.DataFrame(new_rows)

def merge_text_rows(df):
    i = 1
    while i < len(df):
        if not df.iloc[i]['text'][0].isupper():
            df.at[i - 1, 'text'] += ' ' + df.iloc[i]['text']
            df = df.drop(df.index[i])
            df = df.reset_index(drop=True)
        else:
            i += 1
    return df

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to get the part of speech tag for lemmatization
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Function to lemmatize a sentence
def lemmatize_sentence(sentence):
    words = word_tokenize(sentence)
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]
    return ' '.join(lemmatized_words)


    
def filter_by_topic(df, topic, color_scheme):

    df_filtered = df[df['topic'] == topic]
    df_filtered['party_date_count'] = df_filtered.groupby(['party', 'date'])['date'].transform('count')
    df_pivot = df_filtered.pivot_table(index='date', columns='party', values='party_date_count', aggfunc='first').fillna(0)
    # df_pivot.index = pd.to_datetime(df_pivot.index).year

    # Plot the data
    plt.figure(figsize=(12, 8))
    for party in df_pivot.columns:
        # plt.plot(df_pivot.index, df_pivot[party], label=party)
        # if df_pivot[party].max() > 0:
        plt.plot(df_pivot.index, df_pivot[party], label=party, color=[c/255 for c in color_scheme[party]])
        plt.gca().set_facecolor((0.9, 0.95, 1))  # Set background color to light grey/blue

    
    plt.xlabel('Year')
    
    plt.ylabel('Party Date Count')
    plt.title('Party Date Count Over Time for Topic {}'.format(topic))
    plt.legend()
    plt.show()
    return df_pivot




















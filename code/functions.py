

''' file contains supporting functions to manipulate datasets or help plotting '''

import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt

import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from wordcloud import WordCloud



def split_text_by_date(df, column="text", split_date=1998):
    new_rows = []
    for index, row in df.iterrows():
        if row['year'] < split_date:
            sentences = nltk.sent_tokenize(row[column])
            for i, sentence in enumerate(sentences):
                # Check if the sentence ends with a digit followed by a period
                if re.search(r'\d+\.$', sentence) and i + 1 < len(sentences) and sentences[i + 1][0].isupper():
                    sentence += ' ' + sentences[i + 1]
                    sentences[i + 1] = sentence
                    continue
                # Disregard abbreviations like "bzw.", "z.B.", "u.a."
                if re.search(r'\b(bzw|z\.B|u\.a)\.$', sentence):
                    new_row = row.copy()
                    new_row[column] = sentence
                    new_rows.append(new_row)
                    continue
                new_row = row.copy()
                new_row[column] = sentence
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

def process_manifesto_data(df):
    """
    Processes the manifesto dataframe by performing various operations such as counting occurrences,
    mapping values, filtering rows, and merging text rows.

    Parameters:
    df (pd.DataFrame): The dataframe to process.

    Returns:
    pd.DataFrame: The processed dataframe.
    """
    new_df = split_text_by_date(df)
    # Count occurrences of distinct values in the party column
    party_counts = df["party"].value_counts()

    # Add a line that contains this count per party
    new_df["doc size of party"] = new_df["party"].map(party_counts)
    new_df["cmp_code"] = pd.to_numeric(new_df["cmp_code"], errors='coerce')
    new_df.index = range(1, len(new_df) + 1)
    
    # Remove all punctuation from the 'text' column
    new_df["text"] = new_df["text"].str.replace(r'[^\w\s]', '', regex=True)  # Delete entries that do not contain words in the text column
    new_df = new_df[new_df['text'].str.contains(r'\b\w+\b', na=False)]

    # Ensure the 'text' column contains only strings and handle NaN values
    new_df["text"] = new_df["text"].astype(str).fillna("")

    # Merge text rows
    new_df = merge_text_rows(new_df)

    new_df = split_text_by_date(new_df)

    return new_df

def apply_labels(df, topic, code, policy, country="both"):
    df_labeled = df[
    (df["topic"].isin(topic)) & 
    (df["cmp_code"].isin(code))
    ]

    if country != "both":
        new_df_labeled = df_labeled.loc[df_labeled["country"] == country]
        new_df_labeled["label"] = policy
        return new_df_labeled
    else:
        df_labeled["label"] = policy
        return df_labeled

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
    df_filtered['party_date_count'] = df_filtered.groupby(['party', 'year'])['year'].transform('count')
    df_pivot = df_filtered.pivot_table(index='year', columns='party', values='party_date_count', aggfunc='first').fillna(0)
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



def generate_wordcloud(dataframe, 
                       topic_ids, 
                       cmp_codes,  
                       width=800, 
                       height=400, 
                       background_color="white"):
    """
    Generate and display a word cloud for the specified topic and cmp codes.

    Parameters:
    dataframe (pd.DataFrame): The input dataframe containing text data.
    topic_ids (list): List of topic IDs to filter the dataframe.
    cmp_codes (list): List of cmp codes to filter the dataframe.
    stopwords_set (set): Set of stopwords to exclude from the word cloud.
    width (int): Width of the word cloud image. Default is 800.
    height (int): Height of the word cloud image. Default is 400.
    background_color (str): Background color of the word cloud. Default is "white".

    Returns:
    None: Displays the generated word cloud.
    """
    # Filter the dataframe for the specified conditions
    # filtered_data = apply_labels(dataframe, topic_ids, cmp_codes, 0)
    filtered_data = dataframe[(dataframe["cmp_code"] == cmp_codes) & (dataframe["topic"] == topic_ids)]

    # Combine all text into a single string
    combined_text = " ".join(filtered_data["text"].dropna())

    stopwords_set = set(stopwords.words('german'))  # Generate the word cloud
    wordcloud = WordCloud(width=width, height=height, background_color=background_color, stopwords=stopwords_set).generate(combined_text)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def count_party_positions(df, col, dict_colors):
    austerity = df[(df[col] == -1)]
    expenditure = df[(df[col] == 1)]
    # Count the number of statements per party
    austerity_distribution = austerity.groupby("party").size() / austerity.groupby("party")["doc size of party"].first()
    expenditure_distribution = expenditure.groupby("party").size() / expenditure.groupby("party")["doc size of party"].first()

    merged_distribution = pd.merge(
    austerity_distribution.rename("Austerity Labels"),
    expenditure_distribution.rename("Expenditure Labels"),
    left_index=True,
    right_index=True,
    how="outer"  # Use outer join to ensure no rows are dropped
    ).fillna(0) * 100

    merged_distribution = merged_distribution.reindex(dict_colors.keys())
    
    return merged_distribution

def plot_cmp_code_heatmap(df, category, log):
    """
    This function processes the given DataFrame to create a heatmap of cmp_code counts over time.

    Parameters:
    df_manifesto (pd.DataFrame): The input DataFrame containing 'cmp_code', 'date', and 'text' columns.

    Returns:
    None: Displays a heatmap.
    """
    # Ensure 'cmp_code' is numeric and drop rows with NaN values in 'cmp_code'
    df['cmp_code'] = pd.to_numeric(df['cmp_code'], errors='coerce')
    df_cleaned = df.dropna(subset=['cmp_code'])

    # Drop rows with invalid 'date' values
    df_cleaned = df_cleaned.dropna(subset=['year'])

    # Filter rows where the first digit of 'cmp_code' equals zero
    if type(category) == str:
        df_cleaned = df_cleaned[df_cleaned['cmp_code'].apply(lambda x: any(str(x).startswith(str(cat)) for cat in category))]
    elif type(category) == list:
        df_cleaned = df_cleaned[df_cleaned['cmp_code'].apply(lambda x: any(str(x).startswith(str(cat)) for cat in category))]

    # Create a pivot table with 'date' on x-axis and 'cmp_code' on y-axis
    df_cmp_code_count = df_cleaned.pivot_table(index='cmp_code', columns='year', values='text', aggfunc='count', fill_value=0)

    # Group rows by the integer part of 'cmp_code' and sum their values
    df_cmp_code_count = df_cmp_code_count.groupby(df_cmp_code_count.index.astype(int)).sum()
    

    # Calculate the logarithm of the values in df_cmp_code_count, adding 1 to avoid log(0)
    df_cmp_code_count_log = df_cmp_code_count.applymap(lambda x: np.log(x + 1))


    if log:
        return df_cmp_code_count_log
    else:
        return df_cmp_code_count

def restructure_years(df, df2):
    heatmap = df.copy()
    # Merge columns and sum their values
    heatmap['1998/99'] = heatmap[1998] + heatmap[1999]
    heatmap['2005/06'] = heatmap[2005] + heatmap[2006]
    if 2008 in heatmap.columns:
        heatmap['2008/09'] = heatmap[2008] + heatmap[2009]
    heatmap['2005/06'] = heatmap[2005] + heatmap[2006]
    heatmap['2019/21'] = heatmap[2019] + heatmap[2021]

    # Reorder the columns
    heatmap = heatmap[['1998/99', 2002, '2005/06', '2008/09', 2013, 2017, '2019/21']]
    heatmap = heatmap.merge(df2[['code', 'title', 'description_md']], 
                                  left_on='cmp_code', 
                                  right_on='code', 
                                  how='left')
    heatmap.iloc[:, :-3] = heatmap.iloc[:, :-3].applymap(lambda x: np.log(x + 1))    
    return heatmap

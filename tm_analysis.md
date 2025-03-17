Import all important stuff


```python
import ssl

from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
from bertopic.representation import KeyBERTInspired
import pandas as pd
import nltk
from datetime import datetime
from textblob import TextBlob
from datasets import load_dataset

import seaborn as sns
import matplotlib.pyplot as plt
import pickle

import importlib
import functions


import nbconvert

nltk.download('punkt_tab')
```


```python
from functions import *
importlib.reload(functions)

```

Load all data 


```python
parties = ["CDU", "SPD", "FDP", "AFD", "LEFT", "GREENS"]
file_path = "C:/Users/Jacob/OneDrive/uni/MA WiSoz/Semester III/Computational Social Sciences/foPra/data/"

```


```python
# Load the DataFrame from a pickle file
df_manifesto = pd.read_pickle('Manifesto_final.pkl') 
```


```python

```


```python
cmp_categories = pd.read_csv(file_path  + "cmp_categories.csv")
```


```python
#topic_model = BERTopic.load("bertopic_model.pkl")
topic_model = BERTopic.load(file_path[:-5])

```


```python
tm_df = topic_model.get_topic_info()
```


```python
# tm_200 = topic_model.reduce_topics(df_manifesto["text"], nr_topics=200)
# tm_400 = topic_model.reduce_topics(df_manifesto["text"], nr_topics=400)

```


```python
with open("topics_over_time.pkl", "rb") as f:
    topics_over_time = pickle.load(f)
```


```python
# Load the topics from a pickle file
topics = pd.read_pickle('topics.pkl')

# Load the probabilities from a pickle file
probs = pd.read_pickle('probs.pkl')

#df_manifesto["topic"] = topics

```


```python
economy = limit_to_cmp(df_manifesto, 400, 450)

```


```python

```

Next: filter topics and codes to automatically code the topics.


```python
topics_filter = get_topics(df_manifesto, cmp_categories)

```


```python
list_topic_409 = topics_filter["Keynesian Demand Management"]
df_subset_409 = tm_df[tm_df['Topic'].isin(list_topic_409)]
df_subset_409
```


```python
extract_topics(self, topic_model, documents, c_tf_idf, topics)

```

Inspect the data


```python
topic_model.visualize_topics()
```


```python
len(topic_model.get_topic(0))
```

Group and Compare Topics Across Parties
Your goal is to compare how political parties talk about different topics. You can:

Analyze topic distribution per party:


```python
party_topic_counts = df_manifesto.groupby(["party", "topic"]).size().unstack().fillna(0)
party_topic_counts = party_topic_counts.div(df_manifesto.groupby("party")["party_count"].first(), axis=0)
party_topic_counts


```


```python
# Call the function
ptc_top3 = plot_top_topics(party_topic_counts, 5)
ptc_top3
```


```python
# Call the function with topic number 13
plot_topic_n_counts(party_topic_counts, 2)
```

Cluster topics into categories (e.g., Economy, Environment, Social Policy) using manual labeling or embeddings.

Analyze sentiment per topic & party (to see how parties frame topics differently):

- needs to be filtered for topic


```python
df_manifesto["sentiment"] = df_manifesto["text"].apply(lambda x: TextBlob(x).sentiment.polarity)

```

Visualize Topic Evolution
Overlay different parties on a timeline to compare their topic distributions.
Heatmaps to show intensity of topics across parties and time


```python
pivot = df_manifesto.pivot_table(index="date", columns="topic", values="party", aggfunc="count")
sns.heatmap(pivot.fillna(0), cmap="coolwarm")
plt.show()
```


```python
topics_docs =pd.DataFrame({"topic": topics, "documents": df_manifesto["text"]})
```


```python
# Filter the topics_over_time DataFrame to include only the top 10 topics
top_10_topics = topics_over_time.groupby('Topic').size().nlargest(10).index
filtered_topics_over_time = topics_over_time[topics_over_time['Topic'].isin(top_10_topics)]

# Visualisiere die Themenentwicklung
topic_model.visualize_topics_over_time(topics_over_time , topics=[1,2,3,4,5,6,7,8,9,10]
)
```


```python
topic_model.visualize_heatmap()

```


```python
representation_model = KeyBERTInspired()
topic_model = BERTopic(representation_model=representation_model)
```




```python

ssl._create_default_https_context = ssl._create_unverified_context


#docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']



topic_model = BERTopic()
topics, probs = topic_model.fit_transform(df_manifesto["text"])

```

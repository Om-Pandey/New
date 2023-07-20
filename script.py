import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

# Assume you have a DataFrame named 'df' with columns 'sentence' and 'cluster_label'
# Replace 'sentence' and 'cluster_label' with your actual column names

# Step 1: Clean the sentences
def clean_sentence(sentence):
    # Implement your sentence cleaning function here
    # Remove punctuation, numbers, and Unicode characters, and return the cleaned sentence
    # You can reuse the `clean_sentence` function from the previous script

# Apply the clean_sentence function to the 'sentence' column
df['cleaned_sentence'] = df['sentence'].apply(clean_sentence)

# Step 2: Combine all documents in a cluster into a single document (representing the cluster)
cluster_docs = df.groupby('cluster_label')['cleaned_sentence'].apply(lambda x: ' '.join(x))

# Step 3: Create a bag-of-words representation for the clusters
# Initialize the CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the CountVectorizer on the cluster-level documents
cluster_bow_matrix = vectorizer.fit_transform(cluster_docs)

# Step 4: L1-normalize the bag-of-words representation
cluster_bow_normalized = normalize(cluster_bow_matrix, norm='l1', axis=1)

# Step 5: Modified version of TF-IDF (class-based tf-idf)
# Calculate the term frequency (TF)
tf = cluster_bow_normalized

# Calculate the inverse document frequency (IDF)
# Compute the average number of words per cluster
average_words_per_cluster = cluster_bow_matrix.sum(axis=1).mean()
# Compute the frequency of each word across all clusters
word_frequencies = cluster_bow_matrix.sum(axis=0)
idf = np.log(1 + average_words_per_cluster / (1 + word_frequencies))

# Step 6: Calculate importance scores per word in each class
tfidf = tf * idf

# Convert the TF-IDF scores to a DataFrame for better visualization
tfidf_df = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names_out())
tfidf_df.insert(0, 'Cluster', cluster_docs.index)  # Add 'Cluster' column with cluster labels

# Step 7: Get the top 5 topics per cluster
top_topics_per_cluster = {}
for cluster_id in range(len(cluster_docs)):
    cluster_tfidf_scores = tfidf_df.iloc[cluster_id, 1:]
    top_words_indices = cluster_tfidf_scores.argsort()[-5:][::-1]  # Get the top 5 words
    top_words = [vectorizer.get_feature_names_out()[idx] for idx in top_words_indices]
    top_topics_per_cluster[cluster_id] = top_words

# Print the top 5 topics per cluster
for cluster_id, top_words in top_topics_per_cluster.items():
    print(f"Cluster {cluster_id + 1} - Top 5 Topics: {' '.join(top_words)}")

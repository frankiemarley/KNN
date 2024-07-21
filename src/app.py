import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import sqlite3
import pandas as pd

url = 'https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_movies.csv'
data_movies = pd.read_csv(url)
url = 'https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_credits.csv'
data_credits = pd.read_csv(url)

print(data_movies)
print(data_credits)

# # Print the columns of the DataFrames
# print("data_movies columns:", data_movies.columns)
# print("data_credits columns:", data_credits.columns)

# # Ensure 'id' column exists in data_movies
# if 'id' not in data_movies.columns:
#     print("Error: 'id' column not found in data_movies")
# else:
#     print("'id' column found in data_movies")

# # Ensure 'title' column exists in both DataFrames
# if 'title' not in data_movies.columns or 'title' not in data_credits.columns:
#     print("Error: 'title' column not found in one or both DataFrames")
# else:
#     print("'title' column found in both DataFrames")

# Connect to SQLite database (or create it)
conn = sqlite3.connect('movies.db')

# Store DataFrames in separate tables
data_movies.to_sql('table_movies', conn, if_exists='replace', index=False)
data_credits.to_sql('table_credits', conn, if_exists='replace', index=False)

# SQL query to join the tables on the 'title' column
query = '''
SELECT table_movies.id AS movie_id, table_movies.title, table_movies.overview, table_movies.genres, table_movies.keywords,
       table_credits.cast, table_credits.crew
FROM table_movies
JOIN table_credits ON table_movies.title = table_credits.title
'''

# Execute the query and load the result into a DataFrame
try:
    unified_df = pd.read_sql_query(query, conn)
    print("Unified DataFrame created successfully")
    # print(unified_df)
except Exception as e:
    print("Error executing query:", e)

# Close the database connection
conn.close()

# Function to extract names from JSON
def extract_names(json_str):
    try:
        items = json.loads(json_str)
        return [item['name'].replace(" ", "") for item in items]
    except (TypeError, json.JSONDecodeError):
        return []

# Function to extract first three cast members
def extract_first_three_cast(json_str):
    try:
        items = json.loads(json_str)
        return [item['name'].replace(" ", "") for item in items[:3]]
    except (TypeError, json.JSONDecodeError):
        return []

# Function to extract director's name
def extract_director(json_str):
    try:
        items = json.loads(json_str)
        for item in items:
            if item['job'] == 'Director':
                return item['name'].replace(" ", "")
        return ""
    except (TypeError, json.JSONDecodeError):
        return ""

# Process columns
unified_df['genres'] = unified_df['genres'].apply(extract_names)
unified_df['keywords'] = unified_df['keywords'].apply(extract_names)
unified_df['cast'] = unified_df['cast'].apply(extract_first_three_cast)
unified_df['crew'] = unified_df['crew'].apply(extract_director)
unified_df['overview'] = unified_df['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])

# Combine columns into 'tags'
unified_df['tags'] = unified_df.apply(lambda row: row['genres'] + row['keywords'] + row['cast'] + [row['crew']] + row['overview'], axis=1)

# Convert 'tags' to a single string with elements separated by commas, then replace commas with blanks
unified_df['tags'] = unified_df['tags'].apply(lambda x: ' '.join(x).replace(",", ""))

# Final DataFrame
final_df = unified_df[['movie_id', 'title', 'tags']]

# print(final_df.head())
# print(final_df["tags"][1])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Vectorize the 'tags' column using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(final_df['tags'])

# Calculate the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations based on title
def recommend(movie):
    try:
        # Get the index of the movie that matches the title
        movie_index = final_df[final_df["title"] == movie].index[0]

        # Get the pairwise similarity scores of all movies with that movie
        distances = cosine_sim[movie_index]

        # Sort the movies based on the similarity scores
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        # Print the titles of the 5 most similar movies
        print(f"Movies similar to '{movie}':")
        for i in movie_list:
            print(final_df.iloc[i[0]].title)
    except IndexError:
        print(f"Movie '{movie}' not found in the database.")

# Test the recommendation system
recommend("Batman")


# Vectorize the 'tags' column using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(final_df['tags'])

# Calculate the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations based on title
def recommend(movie, final_df=final_df, cosine_sim=cosine_sim):
    try:
        movie_index = final_df[final_df['title'] == movie].index[0]
        distances = cosine_sim[movie_index]
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        return [final_df.iloc[i[0]].title for i in movie_list]
    except IndexError:
        return []

# Test the recommendation system
recommend("Batman")


import numpy as np
from sklearn.model_selection import train_test_split

def precision_at_k(y_true, y_pred, k):
    """
    Precision at k
    """
    k = min(k, len(y_pred))
    return len(set(y_true) & set(y_pred[:k])) / k

def recall_at_k(y_true, y_pred, k):
    """
    Recall at k
    """
    k = min(k, len(y_pred))
    return len(set(y_true) & set(y_pred[:k])) / len(y_true)

def ndcg_at_k(y_true, y_pred, k):
    """
    Normalized Discounted Cumulative Gain
    """
    k = min(k, len(y_pred))
    dcg = sum((1 / np.log2(i + 2)) for i, item in enumerate(y_pred[:k]) if item in y_true)
    idcg = sum((1 / np.log2(i + 2)) for i in range(min(k, len(y_true))))
    return dcg / idcg if idcg > 0 else 0

def evaluate_recommendations(final_df, cosine_sim, test_data, k=5):
    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    for _, row in test_data.iterrows():
        movie = row['title']
        true_tags = set(row['tags'].split())
        
        recommendations = recommend(movie, final_df, cosine_sim)
        
        recommended_tags = set()
        for rec in recommendations:
            rec_tags = set(final_df[final_df['title'] == rec]['tags'].iloc[0].split())
            recommended_tags.update(rec_tags)
        
        precision_scores.append(precision_at_k(true_tags, list(recommended_tags), k))
        recall_scores.append(recall_at_k(true_tags, list(recommended_tags), k))
        ndcg_scores.append(ndcg_at_k(list(true_tags), list(recommended_tags), k))

    print(f"Mean Precision@{k}: {np.mean(precision_scores):.4f}")
    print(f"Mean Recall@{k}: {np.mean(recall_scores):.4f}")
    print(f"Mean NDCG@{k}: {np.mean(ndcg_scores):.4f}")

# Split the data
train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=42)

# Evaluate the model
evaluate_recommendations(final_df, cosine_sim, test_df)
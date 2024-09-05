import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.filterwarnings('ignore')

def load_data(ratings_file_path, movies_file_path):
    """
    Loads and returns datasets for ratings and movie features from specified file paths.
    """
    dtype_ratings = {'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'}
    dtype_movies = {'movieId': 'int32'}

    df_ratings = pd.read_csv(ratings_file_path, dtype=dtype_ratings)
    df_movies = pd.read_csv(movies_file_path, dtype=dtype_movies)
    
    return df_ratings, df_movies

def prepare_data(df_ratings, df_movies):
    """
    Prepares data by aligning and scaling rating matrix.
    """
    # Ensure 'movieId' columns are of type int
    df_ratings['movieId'] = df_ratings['movieId'].astype('int')
    df_movies['movieId'] = df_movies['movieId'].astype('int')

    # Align movie features to training data
    df_movies_aligned = df_movies[df_movies['movieId'].isin(df_ratings['movieId'].unique())]
    df_movies_aligned = df_movies_aligned.dropna(subset=['genres'])

    # Convert user and movie IDs to categorical types
    user_categories = pd.Categorical(df_ratings['userId'])
    item_categories = pd.Categorical(df_ratings['movieId'])

    # Convert user and item IDs to numerical codes
    user_ids = user_categories.codes
    item_ids = item_categories.codes

    # Create rating matrix
    rating_matrix = np.zeros((user_categories.categories.size, item_categories.categories.size))
    rating_matrix[user_ids, item_ids] = df_ratings['rating']

    # Scale the rating matrix
    scaler = MinMaxScaler(feature_range=(0.5, 5))
    rating_matrix_scaled = scaler.fit_transform(rating_matrix)

    return rating_matrix_scaled, df_movies_aligned, user_categories, item_categories

def train_hybrid_model(rating_matrix_scaled, df_movies_aligned):
    """
    Trains the NMF model on the full feature matrix combining rating and genre features.
    """
    # Vectorize movie genres using TF-IDF
    vectorizer = TfidfVectorizer(max_features=100)
    tags_features = vectorizer.fit_transform(df_movies_aligned['genres'].fillna(''))

    # Combine rating matrix and content features
    importance_of_genre = 0.5
    content_weighted_features = tags_features.multiply(importance_of_genre).toarray()
    full_features_matrix = np.hstack([rating_matrix_scaled.T, content_weighted_features]).T

    # NMF model training
    model = NMF(n_components=15, init='nndsvd', max_iter=30, random_state=42)
    W = model.fit_transform(full_features_matrix)
    H = model.components_

    return W, H

def get_top_n_recommendations(user_id, n, W, H, user_categories, item_categories, df_movies, df_ratings):
    """
    Generates top N movie recommendations for a given user based on NMF model predictions,
    excluding movies already rated by the user.

    Parameters
    ----------
    user_id : int
        The user ID for whom recommendations are to be made.
    n : int
        Number of top recommendations to generate.
    W : np.array
        User feature matrix from NMF.
    H : np.array
        Item feature matrix from NMF.
    user_categories : pd.Categorical
        Categorical data of user IDs.
    item_categories : pd.Categorical
        Categorical data of movie IDs.
    df_movies : pd.DataFrame
        DataFrame containing movie information.
    df_ratings : pd.DataFrame
        DataFrame containing user ratings.

    Returns
    -------
    pd.DataFrame
        DataFrame containing top N recommended movies with columns: movieId, title, and genres.
    """
    if user_id not in user_categories.categories:
        return pd.DataFrame()

    # Predict ratings for the user
    user_idx = user_categories.categories.get_loc(user_id)
    predicted_ratings = np.dot(W[user_idx, :], H)

    # Kullanıcının zaten puanladığı filmleri al
    already_rated_movies = set(df_ratings[df_ratings['userId'] == user_id]['movieId'])

    # Zaten puanlanan filmleri önerilerden çıkar
    recommendations = []
    for idx in np.argsort(predicted_ratings)[::-1]:
        movie_id = item_categories.categories[idx]
        if movie_id not in already_rated_movies:
            recommendations.append(movie_id)
        if len(recommendations) == n:
            break

    # Top N önerileri döndür
    return df_movies[df_movies['movieId'].isin(recommendations)][['movieId', 'title', 'genres']]

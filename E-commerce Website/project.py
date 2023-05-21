from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Load the movie dataset
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = movies.merge(credits, on='id')

# Remove the columns we don't need
movies = movies[['id', 'title_x', 'overview', 'vote_count', 'vote_average']]

# Clean up the overview column
movies['overview'] = movies['overview'].fillna('')

# Calculate the mean rating
C = movies['vote_average'].mean()

# Calculate the minimum number of votes required to be considered
m = movies['vote_count'].quantile(0.9)

# Filter out the movies that don't meet the minimum number of votes requirement
q_movies = movies.copy().loc[movies['vote_count'] >= m]

# Calculate the weighted rating for each movie
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

# Sort the movies by their scores in descending order
q_movies = q_movies.sort_values('score', ascending=False)

# Create a TF-IDF matrix for the movie overviews
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(q_movies['overview'])

# Calculate the cosine similarity between each pair of movies
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a Pandas series with the index being the movie title and the value being the index of the movie in the dataset
indices = pd.Series(q_movies.index, index=q_movies['title_x'])

# Define a function to get movie recommendations based on user input
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return q_movies['title_x'].iloc[movie_indices]

# Define a route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Define a route for the recommendation page
@app.route('/recommendations', methods=['POST'])
def recommendations():
    # Get the user input from the form
    movie_title = request.form['movie_title']
    # Get the movie recommendations
    recommended_movies = get_recommendations(movie_title)
    # Pass the recommended movies to the template
    return render_template('recommendations.html', movie_title=movie_title, recommended_movies=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)

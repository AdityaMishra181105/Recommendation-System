import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Create a simple mock dataset of movies
data = {
    'movie_id': [1, 2, 3, 4, 5, 6],
    'title': [
        'The Dark Knight', 
        'Inception', 
        'Toy Story', 
        'The Hangover', 
        'Interstellar', 
        'Superbad'
    ],
    'genres': [
        'Action Action Thriller Dark',
        'Action Sci-Fi Thriller Sci-Fi',
        'Animation Children Comedy',
        'Comedy Romance',
        'Sci-Fi Adventure Drama Sci-Fi',
        'Comedy Romance'
    ]
}

df = pd.DataFrame(data)

# 2. Convert genres text into a matrix of token counts
cv = CountVectorizer()
count_matrix = cv.fit_transform(df['genres'])

# 3. Compute the Cosine Similarity matrix based on the count_matrix
similarity = cosine_similarity(count_matrix)

# 4. Recommendation Function
def get_recommendations(movie_title, similarity_matrix, dataframe):
    # Find the index of the movie that matches the title
    try:
        movie_idx = dataframe[dataframe['title'].str.lower() == movie_title.lower()].index[0]
    except IndexError:
        return f"Movie '{movie_title}' not found in the database."
    
    # Get pairwise similarity scores of all movies with that movie
    similarity_scores = list(enumerate(similarity_matrix[movie_idx]))
    
    # Sort the movies based on the similarity scores in descending order
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get top 2 most similar movies (excluding the movie itself)
    recommended_indices = [score[0] for score in sorted_scores if score[0] != movie_idx][:2]
    
    # Return titles
    return dataframe['title'].iloc[recommended_indices].tolist()

# --- Test the system ---
if __name__ == "__main__":
    print("--- Movie Recommendation System ---")
    user_movie = "Inception"
    
    print(f"\nBecause you watched '{user_movie}', you might also like:")
    recommendations = get_recommendations(user_movie, similarity, df)
    
    if isinstance(recommendations, list):
        for movie in recommendations:
            print(f"- {movie}")
    else:
        print(recommendations)
# Movie Recommendation System

A machine learning-based movie recommendation engine that uses **TF-IDF vectorization** and **content-based filtering** to suggest similar movies based on user selection.

## Main file is 
movie_recommendation.ipynb

## Machine Learning Approach

### Algorithm: Content-Based Filtering with TF-IDF

1. **TF-IDF Vectorization**: Converts movie descriptions and metadata into numerical vectors
2. **Cosine Similarity**: Measures similarity between movies by calculating cosine distance
3. **Ranking**: Returns top-N most similar movies as recommendations

### Why TF-IDF?
- Captures importance of terms in movie descriptions
- Efficient and fast for large datasets
- Works well for content-based recommendations
- Balances term frequency and document specificity

## Dataset

- **Source**: `movies_metadata.csv`
- **Features used**: Movie titles, descriptions, genres, cast
- **Preprocessing**: Text vectorization and similarity indexing

## Technology Stack

- **Python 3.8+**
- **Pandas** - Data processing
- **Scikit-learn** - TF-IDF and similarity calculations
- **FastAPI** - Backend API
- **Streamlit** - Interactive web interface

## Installation & Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set TMDB API key in .env file
echo "TMDB_API_KEY = 1216421e57246844b54678a782049f26" > .env
```

## Running the System

```bash
# Terminal 1: Start ML backend
python main.py

# Terminal 2: Start interactive frontend
streamlit run app.py
```

## How It Works

```
User selects movie
    ↓
TF-IDF vector lookup for that movie
    ↓
Calculate cosine similarity with all movies
    ↓
Sort by similarity score
    ↓
Return top N recommendations
```

## Model Files (Cached)

- `tfidf.pkl` - Fitted TF-IDF vectorizer
- `tfidf_matrix.pkl` - Pre-computed TF-IDF matrix for all movies
- `indices.pkl` - Movie title to index mapping
- `df.pkl` - Movie metadata dataframe

## Results

Given a movie, the system returns recommendations ranked by similarity score (0-1).

**Example**: Select "The Shawshank Redemption" → Get similar drama/thriller movies with high relevance scores

## Future ML Enhancements

- Hybrid filtering (content + collaborative)
- Deep learning embeddings (Word2Vec, BERT)
- User-based collaborative filtering
- Matrix factorization (SVD)

---

**Built with machine learning fundamentals in mind** 🤖

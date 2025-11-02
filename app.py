"""
Flask Application for Ebuss Recommendation System
This application provides a web interface to get product recommendations based on user input
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Import NLTK dependencies
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

# Download NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

# Initialize the Flask app
app = Flask(__name__)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load models and data
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading models...")
print(f"Models directory: {MODELS_DIR}")

# Initialize all as None
sentiment_model = None
tfidf_vectorizer = None
recommendation_ratings = None
product_mapping = None
user_mapping = None
model_config = None
df = None

try:
    print("Loading sentiment_model.pkl...")
    with open(os.path.join(MODELS_DIR, 'sentiment_model.pkl'), 'rb') as f:
        sentiment_model = pickle.load(f)
    print("Sentiment model loaded successfully")
    
    print("Loading tfidf_vectorizer.pkl...")
    with open(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    print("TF-IDF vectorizer loaded successfully")
    
    print("Loading recommendation_ratings.pkl...")
    with open(os.path.join(MODELS_DIR, 'recommendation_ratings.pkl'), 'rb') as f:
        recommendation_ratings = pickle.load(f)
    print("Recommendation ratings loaded successfully")
    
    print("Loading product_mapping.pkl...")
    with open(os.path.join(MODELS_DIR, 'product_mapping.pkl'), 'rb') as f:
        product_mapping = pickle.load(f)
    print("Product mapping loaded successfully")
    
    print("Loading user_mapping.pkl...")
    with open(os.path.join(MODELS_DIR, 'user_mapping.pkl'), 'rb') as f:
        user_mapping = pickle.load(f)
    print("User mapping loaded successfully")
    
    print("Loading model_config.pkl...")
    with open(os.path.join(MODELS_DIR, 'model_config.pkl'), 'rb') as f:
        model_config = pickle.load(f)
    print("Model config loaded successfully")
    
    print("Loading sample30.csv...")
    df = pd.read_csv(os.path.join(MODELS_DIR, 'sample30.csv'))
    # Create product_id and user_id columns if they don't exist
    if 'product_id' not in df.columns:
        df['product_id'] = df.groupby('name').ngroup()
    if 'user_id' not in df.columns:
        df['user_id'] = df.groupby('reviews_username').ngroup()
    print("Main dataset loaded successfully")
    
    print("All models loaded successfully!")
    
except Exception as e:
    import traceback
    print(f"ERROR loading models: {str(e)}")
    print(f"Traceback: {traceback.format_exc()}")
    # Keep variables as None


def clean_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)


def recommend_products_with_sentiment(username, n_recommendations=20, top_n=5):
    """
    Recommend products by combining recommendations and sentiment analysis
    """
    try:
        # Get user ID from username
        user_row = user_mapping[user_mapping['reviews_username'] == username]
        
        if user_row.empty:
            return None, f"User '{username}' not found in the dataset"
        
        user_id = user_row['user_id'].iloc[0]
        
        # Check if user_id exists in recommendation_ratings
        if user_id not in recommendation_ratings.index:
            return None, f"User '{username}' does not have recommendations available"
        
        # Get top N recommendations
        user_recs = recommendation_ratings.loc[user_id].sort_values(ascending=False)[0:n_recommendations]
        
        # Get product IDs for these recommendations
        product_ids = user_recs.index.tolist()
        
        # Get reviews for these products
        product_reviews = df[df['product_id'].isin(product_ids)]
        
        if product_reviews.empty:
            return None, "No reviews found for recommended products"
        
        # Clean text if not already cleaned
        if 'cleaned_reviews' not in product_reviews.columns:
            product_reviews = product_reviews.copy()
            product_reviews['cleaned_reviews'] = product_reviews['reviews_text'].apply(clean_text)
        
        # Remove empty cleaned reviews
        product_reviews = product_reviews[product_reviews['cleaned_reviews'].str.strip() != '']
        
        if product_reviews.empty:
            return None, "No valid reviews found for recommended products"
        
        # Predict sentiments for reviews
        cleaned_texts = product_reviews['cleaned_reviews'].values
        tfidf_features = tfidf_vectorizer.transform(cleaned_texts)
        predicted_sentiments = sentiment_model.predict(tfidf_features)
        
        # Add predictions to reviews
        product_reviews = product_reviews.copy()
        product_reviews['predicted_sentiment'] = predicted_sentiments
        
        # Calculate positive sentiment ratio for each product
        product_sentiment = product_reviews.groupby('product_id')['predicted_sentiment'].agg(['sum', 'count'])
        product_sentiment['positive_ratio'] = product_sentiment['sum'] / product_sentiment['count']
        
        # Sort by positive sentiment ratio and get top N
        top_products = product_sentiment.sort_values('positive_ratio', ascending=False).head(top_n)
        
        # Get product details
        final_recommendations = []
        for product_id, row in top_products.iterrows():
            # Convert row (Series) to dict for easier access
            row_dict = row.to_dict()
            product_info = product_mapping[product_mapping['product_id'] == product_id]
            if not product_info.empty:
                product_name = product_info['name'].iloc[0]
                brand = product_info['brand'].iloc[0]
                categories = product_info['categories'].iloc[0]
                # Access row data correctly from dict
                positive_ratio = float(row_dict.get('positive_ratio', 0.0))
                final_recommendations.append({
                    'product_name': product_name,
                    'brand': brand,
                    'categories': categories,
                    'positive_sentiment_ratio': positive_ratio
                })
        
        if not final_recommendations:
            return None, "No recommendations generated"
        
        return final_recommendations, None
        
    except Exception as e:
        return None, f"Error generating recommendations: {str(e)}"


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    """Handle recommendation requests"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        
        if not username:
            return jsonify({
                'success': False,
                'error': 'Please enter a username'
            }), 400
        
        recommendations, error = recommend_products_with_sentiment(username, n_recommendations=20, top_n=5)
        
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 400
        
        return jsonify({
            'success': True,
            'username': username,
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/get_users', methods=['GET'])
def get_users():
    """Get list of available usernames"""
    try:
        if user_mapping is not None:
            users = user_mapping['reviews_username'].dropna().unique().tolist()
            # Return first 100 users for dropdown
            users = sorted(users)[:100]
            return jsonify({
                'success': True,
                'users': users
            })
        else:
            import traceback
            return jsonify({
                'success': False,
                'error': f'User mapping not available. Check startup logs: {traceback.format_exc()}'
            }), 500
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': f'Error fetching users: {str(e)}. Traceback: {traceback.format_exc()}'
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)


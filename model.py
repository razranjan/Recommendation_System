"""
Model file containing the core ML logic for Ebuss Recommendation System

This file contains:
1. Sentiment Analysis Model (Random Forest)
2. Recommendation System (Item-Based Collaborative Filtering)
3. Integration logic to combine both systems
"""

import pandas as pd
import numpy as np
import re
import pickle
import os

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, pairwise_distances
from sklearn.preprocessing import MinMaxScaler

# Text processing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK data if needed
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


class EbussRecommendationModel:
    """
    Main model class that integrates sentiment analysis with recommendation system
    """
    
    def __init__(self):
        """Initialize the model with components"""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = None
        self.sentiment_model = None
        self.recommendation_ratings = None
        self.product_mapping = None
        self.user_mapping = None
        self.df = None
        
    def clean_text(self, text):
        """
        Clean and preprocess text data
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def train_sentiment_model(self, df):
        """
        Train sentiment analysis model using Random Forest
        
        Args:
            df: DataFrame with 'cleaned_reviews' and 'user_sentiment' columns
            
        Returns:
            Trained model and accuracy score
        """
        print("Training sentiment analysis model...")
        
        # Initialize TF-IDF Vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        # Prepare features
        df_cleaned = df.dropna(subset=['user_sentiment', 'cleaned_reviews'])
        X_tfidf = self.tfidf_vectorizer.fit_transform(df_cleaned['cleaned_reviews'])
        y = df_cleaned['user_sentiment'].map({'Positive': 1, 'Negative': 0})
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest model
        print("Training Random Forest classifier...")
        self.sentiment_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        self.sentiment_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.sentiment_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Sentiment model accuracy: {accuracy:.4f}")
        
        return accuracy
    
    def train_recommendation_system(self, df):
        """
        Train item-based collaborative filtering recommendation system
        
        Args:
            df: DataFrame with user_id, product_id, and reviews_rating columns
            
        Returns:
            Recommendation ratings matrix
        """
        print("Training recommendation system...")
        
        # Create rating matrix
        rating_df = df[['user_id', 'product_id', 'reviews_rating']].copy()
        
        # Split into train and test
        train, test = train_test_split(rating_df, test_size=0.30, random_state=31)
        
        # Create pivot matrix
        df_pivot = train.pivot_table(
            index='user_id',
            columns='product_id',
            values='reviews_rating',
            aggfunc='mean'
        ).fillna(0)
        
        # Create item-based pivot matrix (transpose)
        df_pivot_item = df_pivot.T
        
        # Normalize item ratings
        mean_item = np.nanmean(df_pivot_item, axis=1)
        df_subtracted_item = (df_pivot_item.T - mean_item).T
        
        # Calculate item similarity using cosine
        item_correlation = 1 - pairwise_distances(
            df_subtracted_item.fillna(0),
            metric='cosine'
        )
        item_correlation[np.isnan(item_correlation)] = 0
        item_correlation[item_correlation < 0] = 0
        
        print("Calculating item similarities...")
        
        # Predict ratings
        item_predicted_ratings = np.dot(df_pivot.fillna(0), item_correlation)
        
        # Create dummy train for filtering unrated products
        dummy_train = train.copy()
        dummy_train['rating'] = dummy_train['reviews_rating'].apply(
            lambda x: 0 if x >= 1 else 1
        )
        dummy_train = dummy_train.pivot_table(
            index='user_id',
            columns='product_id',
            values='rating',
            aggfunc='max'
        ).fillna(1)
        
        item_final_rating = np.multiply(item_predicted_ratings, dummy_train)
        
        print("Recommendation system trained successfully!")
        
        self.recommendation_ratings = pd.DataFrame(
            item_final_rating,
            index=df_pivot.index,
            columns=df_pivot.columns
        )
        
        return self.recommendation_ratings
    
    def get_recommendations(self, username, n_recommendations=20, top_n=5):
        """
        Get top N product recommendations for a user based on sentiment analysis
        
        Args:
            username: Username to get recommendations for
            n_recommendations: Number of initial recommendations (default: 20)
            top_n: Final number of recommendations (default: 5)
            
        Returns:
            List of dictionaries with product details and sentiment scores
        """
        try:
            # Get user ID
            user_row = self.user_mapping[self.user_mapping['reviews_username'] == username]
            
            if user_row.empty:
                return None, f"User '{username}' not found in the dataset"
            
            user_id = user_row['user_id'].iloc[0]
            
            # Check if user_id exists in recommendation_ratings
            if user_id not in self.recommendation_ratings.index:
                return None, f"User '{username}' does not have recommendations available"
            
            # Get top N recommendations
            user_recs = self.recommendation_ratings.loc[user_id].sort_values(
                ascending=False
            )[0:n_recommendations]
            
            # Get product IDs
            product_ids = user_recs.index.tolist()
            
            # Get reviews for these products
            product_reviews = self.df[self.df['product_id'].isin(product_ids)]
            
            if product_reviews.empty:
                return None, "No reviews found for recommended products"
            
            # Clean text if needed
            if 'cleaned_reviews' not in product_reviews.columns:
                product_reviews = product_reviews.copy()
                product_reviews['cleaned_reviews'] = product_reviews['reviews_text'].apply(
                    self.clean_text
                )
            
            # Remove empty cleaned reviews
            product_reviews = product_reviews[
                product_reviews['cleaned_reviews'].str.strip() != ''
            ]
            
            if product_reviews.empty:
                return None, "No valid reviews found for recommended products"
            
            # Predict sentiments
            cleaned_texts = product_reviews['cleaned_reviews'].values
            tfidf_features = self.tfidf_vectorizer.transform(cleaned_texts)
            predicted_sentiments = self.sentiment_model.predict(tfidf_features)
            
            # Add predictions
            product_reviews = product_reviews.copy()
            product_reviews['predicted_sentiment'] = predicted_sentiments
            
            # Calculate positive sentiment ratio
            product_sentiment = product_reviews.groupby('product_id')['predicted_sentiment'].agg(['sum', 'count'])
            product_sentiment['positive_ratio'] = product_sentiment['sum'] / product_sentiment['count']
            
            # Sort by positive sentiment ratio
            top_products = product_sentiment.sort_values(
                'positive_ratio',
                ascending=False
            ).head(top_n)
            
            # Get product details
            final_recommendations = []
            for product_id, row in top_products.iterrows():
                # Convert row (Series) to dict
                row_dict = row.to_dict()
                product_info = self.product_mapping[
                    self.product_mapping['product_id'] == product_id
                ]
                
                if not product_info.empty:
                    product_name = product_info['name'].iloc[0]
                    brand = product_info['brand'].iloc[0]
                    categories = product_info['categories'].iloc[0]
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
    
    def save_models(self, save_directory):
        """
        Save all models to pickle files
        
        Args:
            save_directory: Directory path to save models
        """
        print(f"Saving models to {save_directory}...")
        
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        # Save TF-IDF vectorizer
        with open(os.path.join(save_directory, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        print("TF-IDF vectorizer saved")
        
        # Save sentiment model
        with open(os.path.join(save_directory, 'sentiment_model.pkl'), 'wb') as f:
            pickle.dump(self.sentiment_model, f)
        print("Sentiment model saved")
        
        # Save recommendation ratings
        self.recommendation_ratings.to_pickle(
            os.path.join(save_directory, 'recommendation_ratings.pkl')
        )
        print("Recommendation ratings saved")
        
        # Save product mapping
        with open(os.path.join(save_directory, 'product_mapping.pkl'), 'wb') as f:
            pickle.dump(self.product_mapping, f)
        print("Product mapping saved")
        
        # Save user mapping
        with open(os.path.join(save_directory, 'user_mapping.pkl'), 'wb') as f:
            pickle.dump(self.user_mapping, f)
        print("User mapping saved")
    
    def load_models(self, load_directory):
        """
        Load all models from pickle files
        
        Args:
            load_directory: Directory path to load models from
        """
        print(f"Loading models from {load_directory}...")
        
        try:
            # Load TF-IDF vectorizer
            with open(os.path.join(load_directory, 'tfidf_vectorizer.pkl'), 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            print("TF-IDF vectorizer loaded")
            
            # Load sentiment model
            with open(os.path.join(load_directory, 'sentiment_model.pkl'), 'rb') as f:
                self.sentiment_model = pickle.load(f)
            print("Sentiment model loaded")
            
            # Load recommendation ratings
            with open(os.path.join(load_directory, 'recommendation_ratings.pkl'), 'rb') as f:
                self.recommendation_ratings = pd.read_pickle(f)
            print("Recommendation ratings loaded")
            
            # Load product mapping
            with open(os.path.join(load_directory, 'product_mapping.pkl'), 'rb') as f:
                self.product_mapping = pd.read_pickle(f)
            print("Product mapping loaded")
            
            # Load user mapping
            with open(os.path.join(load_directory, 'user_mapping.pkl'), 'rb') as f:
                self.user_mapping = pd.read_pickle(f)
            print("User mapping loaded")
            
            print("All models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise


# Main training function
def train_ebuss_models(data_path, save_path):
    """
    Complete training pipeline for Ebuss Recommendation System
    
    Args:
        data_path: Path to sample30.csv
        save_path: Directory to save trained models
    """
    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Create product and user IDs
    df['product_id'] = df.groupby('name').ngroup()
    df['user_id'] = df.groupby('reviews_username').ngroup()
    
    # Initialize model
    model = EbussRecommendationModel()
    model.df = df
    
    # Clean text
    print("Cleaning text data...")
    df['cleaned_reviews'] = df['reviews_text'].apply(model.clean_text)
    df = df[df['cleaned_reviews'].str.strip() != '']
    
    # Train sentiment model
    sentiment_accuracy = model.train_sentiment_model(df)
    
    # Train recommendation system
    model.train_recommendation_system(df)
    
    # Create mappings
    model.product_mapping = df[['product_id', 'name', 'brand', 'categories']].drop_duplicates()
    model.user_mapping = df[['reviews_username', 'user_id']].drop_duplicates()
    
    # Save models
    model.save_models(save_path)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print(f"Sentiment Model Accuracy: {sentiment_accuracy:.4f}")
    print(f"Recommendation System: Item-Based Collaborative Filtering")
    print("="*70)
    
    return model


if __name__ == "__main__":
    # Example usage for training
    data_path = 'sample30.csv'
    save_path = './models'
    
    model = train_ebuss_models(data_path, save_path)
    
    # Test with a sample user
    print("\nTesting recommendation system...")
    recommendations, error = model.get_recommendations('joshua')
    
    if recommendations:
        print(f"\nTop 5 Recommendations for 'joshua':")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['product_name']}")
            print(f"   Brand: {rec['brand']}")
            print(f"   Positive Sentiment: {rec['positive_sentiment_ratio']:.2%}")


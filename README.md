# Recommendation System

The e-commerce business is quite popular today. Here, you do not need to take orders by going to each customer. A company launches its website to sell the items to the end consumer, and customers can order the products that they require from the same website. Famous examples of such e-commerce companies are Amazon, Flipkart, Myntra, Paytm and Snapdeal.

 

Suppose you are working as a Machine Learning Engineer in an e-commerce company named 'Ebuss'. Ebuss has captured a huge market share in many fields, and it sells the products in various categories such as household essentials, books, personal care products, medicines, cosmetic items, beauty products, electrical appliances, kitchen and dining products and health care products.

 

With the advancement in technology, it is imperative for Ebuss to grow quickly in the e-commerce market to become a major leader in the market because it has to compete with the likes of Amazon, Flipkart, etc., which are already market leaders.

 

As a senior ML Engineer, you are asked to build a model that will improve the recommendations given to the users given their past reviews and ratings. 

 

In order to do this, build a sentiment-based product recommendation system, which includes the following tasks.

Data sourcing and sentiment analysis
Building a recommendation system
Improving the recommendations using the sentiment analysis model
Deploying the end-to-end project with a user interface

## Project Structure

```
Recommendation_System/
├── app.py                          # Flask application backend
├── templates/
│   └── index.html                  # Web interface (HTML)
├── requirements.txt                # Python dependencies
├── Procfile                        # Heroku deployment configuration
├── runtime.txt                     # Python version for Heroku
├── Ebuss_Recommendation_System.ipynb  # Jupyter notebook with ML models
├── sample30.csv                    # Dataset
├── sentiment_model.pkl             # Trained sentiment analysis model
├── tfidf_vectorizer.pkl            # TF-IDF vectorizer
├── recommendation_ratings.pkl      # Recommendation system ratings
├── product_mapping.pkl             # Product ID to name mapping
├── user_mapping.pkl                # User ID to username mapping
└── model_config.pkl                # Model configuration

```

## Features

- **Sentiment Analysis**: Random Forest model with 94.33% accuracy
- **Recommendation System**: Item-based collaborative filtering
- **User Interface**: Modern, responsive web interface
- **Real-time Recommendations**: Get top 5 product recommendations based on user sentiment

## Local Deployment

### Prerequisites

- Python 3.9+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Recommendation_System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

## Heroku Deployment

### Prerequisites

- Heroku account (sign up at https://www.heroku.com)
- Heroku CLI installed
- Git repository initialized

### Deployment Steps

1. Login to Heroku:
```bash
heroku login
```

2. Create a Heroku app:
```bash
heroku create your-app-name
```

3. Push to Heroku:
```bash
git add .
git commit -m "Initial deployment"
git push heroku main
```

4. Open your deployed app:
```bash
heroku open
```

### Important Notes

- The app will automatically download NLTK data on first run
- Ensure all pickle files are included in the deployment
- The app uses gunicorn as the production server
- Heroku will automatically detect Python version from runtime.txt

## Usage

1. Enter a username from the dataset
2. Click "Get Recommendations"
3. View the top 5 product recommendations with:
   - Product name
   - Brand
   - Category
   - Positive sentiment percentage

## Model Performance

- **Best Sentiment Model**: Random Forest (94.33% accuracy)
- **Best Recommendation Model**: Item-Based (RMSE: 2.30)
- **Total Products**: 271
- **Total Users**: 24,908
- **Total Reviews**: 29,993

## Technology Stack

- **Backend**: Flask, Python
- **Machine Learning**: scikit-learn, XGBoost, NLTK
- **Data Processing**: pandas, numpy
- **Deployment**: Heroku, Gunicorn
- **Frontend**: HTML, CSS, JavaScript

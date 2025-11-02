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
├── model.py                        # ML model class with sentiment & recommendation logic
├── app.py                          # Flask application backend
├── templates/
│   └── index.html                  # Web interface (HTML)
├── requirements.txt                # Python dependencies
├── Procfile                        # Deployment configuration
├── runtime.txt                     # Python version specification
├── Ebuss_Recommendation_System.ipynb  # Complete Jupyter notebook
├── sample30.csv                    # Dataset (30,000 reviews)
├── sentiment_model.pkl             # Trained Random Forest model
├── tfidf_vectorizer.pkl            # TF-IDF vectorizer
├── recommendation_ratings.pkl      # Item-based recommendation matrix
├── product_mapping.pkl             # Product metadata
├── user_mapping.pkl                # User metadata
└── model_config.pkl                # Configuration settings

```

## Features

- **Sentiment Analysis**: Random Forest model with 94.57% accuracy
- **Recommendation System**: Item-based collaborative filtering
- **User Interface**: Modern, responsive web interface
- **Real-time Recommendations**: Get top 5 product recommendations based on user sentiment
- **Live Deployment**: Successfully deployed on Railway

## 🌐 Live Demo

**Deployed Application**: [https://ebuss-recommendation.up.railway.app/](https://ebuss-recommendation.up.railway.app/)

Try it out with usernames like: `joshua`, `dorothy w`, `rebecca`

## Local Deployment

### Prerequisites

- Python 3.10+ (required for newer library versions)
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Recommendation_System
```

2. Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to:
```
http://localhost:5000
```

### Note on Library Compatibility

The pickle files were created with specific library versions. If you encounter compatibility errors, ensure you're using Python 3.10+ with the exact versions specified in `requirements.txt`.

## Deployment Options

### Option 1: Railway Deployment (Recommended - Free & Easy)

Railway is the easiest platform for deploying this application.

#### Prerequisites

- Railway account (sign up at https://railway.app - free tier available)
- GitHub repository with your code
- Git repository initialized

#### Deployment Steps

1. **Push to GitHub:**
```bash
git add .
git commit -m "Deploy Flask recommendation system"
git push origin main
```

2. **Deploy to Railway:**
   - Go to Railway dashboard: https://railway.app
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Connect your GitHub account
   - Select your repository

3. **Configure Settings:**
   - Railway auto-detects Flask apps
   - No additional configuration needed for most cases
   - Your app will be available at `https://your-app.railway.app`

4. **Start Command (if needed):**
   - Start Command: `gunicorn app:app`
   - Build Command: `pip install -r requirements.txt`

5. **Customize Your Domain Name:**
   - Go to the **Settings** tab in your Railway project
   - Under **Service Name**, rename your service (e.g., `ebuss-recommendation`)
   - Railway will generate a new domain based on the service name
   - For a custom subdomain like `ebuss_recommendation.railway.app`, ensure the service name matches (Railway converts underscores to hyphens in domains)
   - Your new URL will be available immediately after the name change

That's it! Your app is live!

#### Railway Benefits

- ✅ Free tier available
- ✅ Automatic HTTPS
- ✅ Easy GitHub integration
- ✅ No payment verification required
- ✅ Simple deployment process



Render offers free hosting as well:

1. Sign up at https://render.com
2. Create new Web Service
3. Connect GitHub repository
4. Configure:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`

### Deployment General Notes

- The app will automatically download NLTK data on first run
- Ensure all pickle files are included in deployment
- The app uses gunicorn as the production server
- Python 3.10+ is required for compatibility

## Usage

1. Enter a username from the dataset
2. Click "Get Recommendations"
3. View the top 5 product recommendations with:
   - Product name
   - Brand
   - Category
   - Positive sentiment percentage

## Model Performance

- **Best Sentiment Model**: Random Forest (94.57% accuracy)
- **Best Recommendation Model**: Item-Based (RMSE: 2.30)
- **Total Products**: 271
- **Total Users**: 24,908
- **Total Reviews**: 29,993

## Technology Stack

- **Backend**: Flask 2.1.2, Python 3.10+
- **Machine Learning**: scikit-learn 1.6.1, XGBoost 3.1.1, NLTK 3.7
- **Data Processing**: pandas 2.2.2, numpy 2.0.2
- **Deployment**: Railway (Recommended), Heroku, Render, Gunicorn
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)

## Library Versions

For compatibility with the trained pickle files, use these exact versions:

- Python: 3.10+
- Flask: 2.1.2
- pandas: 2.2.2
- numpy: 2.0.2
- scikit-learn: 1.6.1
- XGBoost: 3.1.1
- NLTK: 3.7
- scipy: >=1.11.0
- joblib: >=1.3.0

See `requirements.txt` for the complete list.

## Troubleshooting

### Pickle Compatibility Errors

If you encounter errors like "No module named 'numpy._core'", this means your library versions don't match the training environment. 

**Solution:** Ensure you have Python 3.10+ and install from `requirements.txt`.
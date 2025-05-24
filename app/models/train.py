import os
import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import mlflow
import mlflow.sklearn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logger.warning(f"NLTK download warning: {str(e)}")

# Set MLflow tracking URI
mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
mlflow.set_experiment("sentiment-analysis")

def preprocess_text(text):
    """Preprocess text for sentiment analysis"""
    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Tokenize
    tokens = word_tokenize(str(text).lower())
    
    # Remove stopwords and lemmatize
    processed_tokens = [
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if token.isalpha() and token not in stop_words
    ]
    
    return ' '.join(processed_tokens)

def load_data(data_path=None):
    """Load and preprocess the dataset"""
    # For demonstration, create a synthetic dataset if no path provided
    if not data_path or not os.path.exists(data_path):
        logger.info("Creating synthetic sentiment dataset for demonstration")
        
        # Create synthetic data
        positive_texts = [
            "I love this product, it's amazing!",
            "The service was excellent and staff very friendly",
            "This is the best movie I've seen in years",
            "I'm extremely satisfied with my purchase",
            "The experience exceeded all my expectations"
        ]
        
        negative_texts = [
            "This was a terrible experience, would not recommend",
            "The product quality is very poor",
            "I'm disappointed with the service I received",
            "This is the worst purchase I've ever made",
            "The customer support was unhelpful and rude"
        ]
        
        neutral_texts = [
            "The product arrived on time",
            "It works as described in the manual",
            "The color is exactly as shown in the picture",
            "I received the order yesterday",
            "The price is average compared to other options"
        ]
        
        # Combine data
        texts = positive_texts + negative_texts + neutral_texts
        labels = [1] * len(positive_texts) + [0] * len(negative_texts) + [0.5] * len(neutral_texts)
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': texts,
            'sentiment': labels
        })
        
        # Save synthetic dataset
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/synthetic_sentiment_data.csv', index=False)
        
        return df
    
    # Load real dataset if path provided
    logger.info(f"Loading sentiment dataset from {data_path}")
    return pd.read_csv(data_path)

def train_model():
    """Train and evaluate the sentiment analysis model"""
    # Start MLflow run
    with mlflow.start_run(run_name=f"sentiment-model-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        # Load data
        df = load_data()
        logger.info(f"Loaded dataset with {len(df)} samples")
        
        # Preprocess text
        logger.info("Preprocessing text data...")
        df['processed_text'] = df['text'].apply(preprocess_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], 
            df['sentiment'], 
            test_size=0.2, 
            random_state=42
        )
        
        # Feature extraction
        logger.info("Extracting features with TF-IDF...")
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train model
        logger.info("Training sentiment analysis model...")
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_vec, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_vec)
        
        # Calculate metrics
        # Convert multi-class to binary for metric calculation
        y_test_binary = [1 if y >= 0.7 else 0 for y in y_test]
        y_pred_binary = [1 if y >= 0.7 else 0 for y in y_pred]
        
        accuracy = accuracy_score(y_test_binary, y_pred_binary)
        precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)
        
        # Log metrics
        logger.info(f"Model metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("vectorizer", "TfidfVectorizer")
        mlflow.log_param("max_features", 5000)
        
        # Save model and vectorizer
        logger.info("Saving model and vectorizer...")
        os.makedirs('models', exist_ok=True)
        
        # Save as pickle files
        with open('models/sentiment_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        with open('models/vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "sentiment_model")
        
        # Create model info file
        model_info = {
            "version": "1.0.0",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1)
            }
        }
        
        # Return model info
        return model_info

if __name__ == "__main__":
    logger.info("Starting sentiment analysis model training")
    model_info = train_model()
    logger.info(f"Model training completed successfully. Version: {model_info['version']}")
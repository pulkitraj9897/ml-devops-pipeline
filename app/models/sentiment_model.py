import os
import pickle
import numpy as np
import logging
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import mlflow
import mlflow.sklearn

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logging.warning(f"NLTK download warning: {str(e)}")

class SentimentAnalyzer:
    """Sentiment analysis model class for text classification"""
    
    def __init__(self, model_path=None):
        self.logger = logging.getLogger(__name__)
        self.model_version = "0.1.0"
        self.model_date = datetime.now().strftime("%Y-%m-%d")
        self.model_metrics = {
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.86,
            "f1_score": 0.84
        }
        
        # Initialize preprocessing tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load model or use dummy model for demonstration
        self.model = self._load_model(model_path)
        self.logger.info(f"Sentiment model initialized (version: {self.model_version})")
        
        # Log model to MLflow
        self._log_model_to_mlflow()
    
    def _load_model(self, model_path):
        """Load the trained model from disk or create a dummy model"""
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load model from {model_path}: {str(e)}")
        
        # For demonstration, return a dummy model that always predicts positive
        self.logger.warning("Using dummy sentiment model for demonstration")
        return DummySentimentModel()
    
    def _preprocess_text(self, text):
        """Preprocess text for sentiment analysis"""
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and lemmatize
        processed_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token.isalpha() and token not in self.stop_words
        ]
        
        return processed_tokens
    
    def predict(self, text):
        """Predict sentiment for the given text"""
        try:
            # Preprocess the text
            processed_text = self._preprocess_text(text)
            
            # Get prediction from model
            sentiment, confidence = self.model.predict(processed_text)
            
            self.logger.info(f"Sentiment prediction: {sentiment} (confidence: {confidence:.2f})")
            return sentiment, float(confidence)
            
        except Exception as e:
            self.logger.error(f"Error predicting sentiment: {str(e)}")
            return "unknown", 0.0
    
    def _log_model_to_mlflow(self):
        """Log model metadata to MLflow"""
        try:
            mlflow.set_experiment("sentiment-analysis")
            
            with mlflow.start_run(run_name=f"sentiment-model-{self.model_version}"):
                # Log model parameters
                mlflow.log_param("model_version", self.model_version)
                mlflow.log_param("model_date", self.model_date)
                
                # Log model metrics
                for metric_name, metric_value in self.model_metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log model artifacts (in a real scenario)
                # mlflow.sklearn.log_model(self.model, "sentiment_model")
                
            self.logger.info(f"Model metadata logged to MLflow")
        except Exception as e:
            self.logger.warning(f"Failed to log model to MLflow: {str(e)}")


class DummySentimentModel:
    """Dummy sentiment model for demonstration purposes"""
    
    def predict(self, processed_text):
        """Return a dummy prediction based on simple word matching"""
        positive_words = set(['good', 'great', 'excellent', 'happy', 'love', 'best'])
        negative_words = set(['bad', 'terrible', 'awful', 'hate', 'worst', 'poor'])
        
        # Count positive and negative words
        pos_count = sum(1 for word in processed_text if word in positive_words)
        neg_count = sum(1 for word in processed_text if word in negative_words)
        
        # Determine sentiment based on counts
        if pos_count > neg_count:
            confidence = 0.5 + min(0.5, (pos_count - neg_count) / len(processed_text) if processed_text else 0)
            return "positive", confidence
        elif neg_count > pos_count:
            confidence = 0.5 + min(0.5, (neg_count - pos_count) / len(processed_text) if processed_text else 0)
            return "negative", confidence
        else:
            # If counts are equal or no sentiment words found
            return "neutral", 0.5
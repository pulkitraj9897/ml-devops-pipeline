from flask import Blueprint, request, jsonify
from models.sentiment_model import SentimentAnalyzer
import logging

logger = logging.getLogger(__name__)
sentiment_blueprint = Blueprint('sentiment', __name__)
sentiment_analyzer = SentimentAnalyzer()

@sentiment_blueprint.route('/predict', methods=['POST'])
def predict_sentiment():
    """Endpoint to predict sentiment from text input"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing required parameter: text'
            }), 400
            
        text = data['text']
        logger.info(f"Received sentiment analysis request for text: {text[:50]}...")
        
        # Get prediction from model
        sentiment, confidence = sentiment_analyzer.predict(text)
        
        return jsonify({
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'model_version': sentiment_analyzer.model_version
        })
        
    except Exception as e:
        logger.error(f"Error processing sentiment request: {str(e)}")
        return jsonify({
            'error': 'Failed to process request',
            'details': str(e)
        }), 500

@sentiment_blueprint.route('/model-info', methods=['GET'])
def model_info():
    """Endpoint to get information about the current model"""
    try:
        return jsonify({
            'model_version': sentiment_analyzer.model_version,
            'model_date': sentiment_analyzer.model_date,
            'model_metrics': sentiment_analyzer.model_metrics
        })
    except Exception as e:
        logger.error(f"Error retrieving model info: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve model information',
            'details': str(e)
        }), 500
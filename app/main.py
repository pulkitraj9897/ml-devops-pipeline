import os
from flask import Flask, jsonify
from api.sentiment_api import sentiment_blueprint
from utils.logging_config import configure_logging
import logging

# Configure application logging
configure_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Register blueprints
app.register_blueprint(sentiment_blueprint, url_prefix='/api/v1')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for the application"""
    return jsonify({
        'status': 'healthy',
        'service': 'sentiment-analysis-api'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting sentiment analysis API server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true')
# ML-DevOps Pipeline Project

This project demonstrates a complete machine learning model deployment pipeline using DevOps practices. It's designed to showcase both machine learning skills and the ability to operationalize ML models in production environments.

## Project Overview

This project implements a sentiment analysis model with a complete CI/CD pipeline for automated testing, containerization, and deployment. The system includes monitoring and model performance tracking capabilities.

## Key Components

1. **Machine Learning Model**: A sentiment analysis model built with scikit-learn and NLTK
2. **API Service**: Flask-based REST API for model serving
3. **CI/CD Pipeline**: GitHub Actions workflow for continuous integration and deployment
4. **Containerization**: Docker for packaging the application
5. **Orchestration**: Kubernetes configuration for deployment
6. **Monitoring**: Prometheus and Grafana for system monitoring
7. **Model Tracking**: MLflow for experiment tracking and model versioning

## Project Structure

```
├── .github/workflows/    # CI/CD pipeline configuration
├── app/                  # Flask application
│   ├── api/              # API endpoints
│   ├── models/           # ML model code
│   └── utils/            # Utility functions
├── data/                 # Training and test data
├── kubernetes/           # Kubernetes deployment files
├── mlflow/               # MLflow configuration
├── monitoring/           # Prometheus and Grafana configuration
├── notebooks/            # Jupyter notebooks for model development
├── tests/                # Unit and integration tests
├── Dockerfile            # Docker configuration
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the development server: `python app/main.py`
4. Access the API at `http://localhost:5000`

## CI/CD Pipeline

The CI/CD pipeline automates the following steps:
1. Run tests on code changes
2. Build and push Docker image
3. Deploy to Kubernetes cluster
4. Monitor application performance

## Model Training and Evaluation

To train the model:
1. Run `python app/models/train.py`
2. View training metrics in MLflow UI

## Deployment

The application can be deployed using:
```
kubectl apply -f kubernetes/deployment.yaml
```

## Monitoring

Access Grafana dashboards at `http://localhost:3000` after deployment.

## License

MIT
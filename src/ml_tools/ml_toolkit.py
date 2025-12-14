"""Machine Learning toolkit for StressGuard chatbot."""

import logging
from typing import Any, Dict, Optional, List
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLToolkit:
    """Machine Learning tools for data analysis and predictions."""
    
    def __init__(self):
        """Initialize ML toolkit."""
        self.models = {}
        self.scalers = {}
        logger.info("ML Toolkit initialized")
        
    def train_classifier(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str = "default_classifier",
        test_size: float = 0.2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a classification model.
        
        Args:
            X: Feature data
            y: Target labels
            model_name: Name to store the model
            test_size: Proportion of data for testing
            **kwargs: Additional parameters for the classifier
            
        Returns:
            Training results including accuracy
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Initialize and train classifier
            clf = RandomForestClassifier(**kwargs)
            clf.fit(X_train, y_train)
            
            # Evaluate
            train_score = clf.score(X_train, y_train)
            test_score = clf.score(X_test, y_test)
            
            # Store model
            self.models[model_name] = clf
            
            logger.info(f"Classifier '{model_name}' trained successfully")
            
            return {
                "success": True,
                "model_name": model_name,
                "train_accuracy": train_score,
                "test_accuracy": test_score,
                "n_features": X.shape[1],
                "n_samples": X.shape[0],
            }
            
        except Exception as e:
            logger.error(f"Error training classifier: {e}")
            return {"success": False, "error": str(e)}
            
    def train_regressor(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str = "default_regressor",
        test_size: float = 0.2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a regression model.
        
        Args:
            X: Feature data
            y: Target values
            model_name: Name to store the model
            test_size: Proportion of data for testing
            **kwargs: Additional parameters for the regressor
            
        Returns:
            Training results including RMSE
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Initialize and train regressor
            reg = RandomForestRegressor(**kwargs)
            reg.fit(X_train, y_train)
            
            # Evaluate
            train_pred = reg.predict(X_train)
            test_pred = reg.predict(X_test)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            # Store model
            self.models[model_name] = reg
            
            logger.info(f"Regressor '{model_name}' trained successfully")
            
            return {
                "success": True,
                "model_name": model_name,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "n_features": X.shape[1],
                "n_samples": X.shape[0],
            }
            
        except Exception as e:
            logger.error(f"Error training regressor: {e}")
            return {"success": False, "error": str(e)}
            
    def cluster_data(
        self,
        X: np.ndarray,
        n_clusters: int = 3,
        model_name: str = "default_clusterer",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform clustering on data.
        
        Args:
            X: Feature data
            n_clusters: Number of clusters
            model_name: Name to store the model
            **kwargs: Additional parameters for KMeans
            
        Returns:
            Clustering results
        """
        try:
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
            labels = kmeans.fit_predict(X)
            
            # Store model
            self.models[model_name] = kmeans
            
            logger.info(f"Clustering '{model_name}' completed successfully")
            
            return {
                "success": True,
                "model_name": model_name,
                "n_clusters": n_clusters,
                "labels": labels.tolist(),
                "inertia": kmeans.inertia_,
                "cluster_centers": kmeans.cluster_centers_.tolist(),
            }
            
        except Exception as e:
            logger.error(f"Error in clustering: {e}")
            return {"success": False, "error": str(e)}
            
    def predict(
        self,
        X: np.ndarray,
        model_name: str = "default_classifier"
    ) -> Dict[str, Any]:
        """
        Make predictions using a trained model.
        
        Args:
            X: Feature data
            model_name: Name of the model to use
            
        Returns:
            Predictions
        """
        try:
            if model_name not in self.models:
                return {
                    "success": False,
                    "error": f"Model '{model_name}' not found"
                }
                
            model = self.models[model_name]
            predictions = model.predict(X)
            
            return {
                "success": True,
                "predictions": predictions.tolist(),
            }
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return {"success": False, "error": str(e)}
            
    def analyze(self, data: Any, task: str = "classification") -> Dict[str, Any]:
        """
        General analysis method for ML tasks.
        
        Args:
            data: Input data
            task: Type of ML task
            
        Returns:
            Analysis results
        """
        return {
            "task": task,
            "message": "ML analysis capability ready",
            "available_models": list(self.models.keys()),
        }

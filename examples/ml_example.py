"""Example: Using ML toolkit."""

import numpy as np
from ml_tools.ml_toolkit import MLToolkit


def main():
    """Run ML toolkit example."""
    print("Initializing ML Toolkit...\n")
    
    ml_toolkit = MLToolkit()
    
    # Example 1: Classification
    print("Example 1: Training a classifier")
    print("-" * 40)
    
    # Generate sample data
    X_classification = np.random.randn(100, 10)
    y_classification = np.random.randint(0, 3, 100)
    
    result = ml_toolkit.train_classifier(
        X_classification,
        y_classification,
        model_name="stress_classifier",
        n_estimators=50
    )
    
    print(f"Success: {result['success']}")
    print(f"Train accuracy: {result['train_accuracy']:.4f}")
    print(f"Test accuracy: {result['test_accuracy']:.4f}\n")
    
    # Example 2: Regression
    print("Example 2: Training a regressor")
    print("-" * 40)
    
    X_regression = np.random.randn(100, 5)
    y_regression = np.random.randn(100)
    
    result = ml_toolkit.train_regressor(
        X_regression,
        y_regression,
        model_name="stress_regressor",
        n_estimators=50
    )
    
    print(f"Success: {result['success']}")
    print(f"Train RMSE: {result['train_rmse']:.4f}")
    print(f"Test RMSE: {result['test_rmse']:.4f}\n")
    
    # Example 3: Clustering
    print("Example 3: Clustering data")
    print("-" * 40)
    
    X_clustering = np.random.randn(100, 3)
    
    result = ml_toolkit.cluster_data(
        X_clustering,
        n_clusters=3,
        model_name="stress_clusters"
    )
    
    print(f"Success: {result['success']}")
    print(f"Number of clusters: {result['n_clusters']}")
    print(f"Inertia: {result['inertia']:.4f}\n")
    
    # Example 4: Making predictions
    print("Example 4: Making predictions")
    print("-" * 40)
    
    X_new = np.random.randn(5, 10)
    result = ml_toolkit.predict(X_new, model_name="stress_classifier")
    
    print(f"Success: {result['success']}")
    print(f"Predictions: {result['predictions']}")


if __name__ == '__main__':
    main()

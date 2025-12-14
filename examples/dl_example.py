"""Example: Using DL toolkit."""

import numpy as np
from dl_tools.dl_toolkit import DLToolkit


def main():
    """Run DL toolkit example."""
    print("Initializing DL Toolkit...\n")
    
    dl_toolkit = DLToolkit()
    
    # Example 1: Create a sequential model
    print("Example 1: Creating a sequential model")
    print("-" * 40)
    
    result = dl_toolkit.create_sequential_model(
        input_shape=(20,),
        output_units=3,
        hidden_layers=[64, 32],
        model_name="stress_predictor"
    )
    
    print(f"Success: {result['success']}")
    print(f"Model name: {result['model_name']}")
    print(f"Total parameters: {result['total_params']:,}\n")
    
    # Example 2: Compile the model
    print("Example 2: Compiling the model")
    print("-" * 40)
    
    result = dl_toolkit.compile_model(
        model_name="stress_predictor",
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print(f"Success: {result['success']}")
    print(f"Optimizer: {result['optimizer']}")
    print(f"Loss: {result['loss']}\n")
    
    # Example 3: Train the model
    print("Example 3: Training the model")
    print("-" * 40)
    
    # Generate sample training data
    X_train = np.random.randn(1000, 20)
    y_train = np.random.randint(0, 3, 1000)
    X_val = np.random.randn(200, 20)
    y_val = np.random.randint(0, 3, 200)
    
    result = dl_toolkit.train_model(
        model_name="stress_predictor",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=5,
        batch_size=32
    )
    
    print(f"Success: {result['success']}")
    print(f"Epochs completed: {result['epochs_completed']}")
    print(f"Final loss: {result['final_loss']:.4f}")
    if result['final_accuracy']:
        print(f"Final accuracy: {result['final_accuracy']:.4f}\n")
    
    # Example 4: Making predictions
    print("Example 4: Making predictions")
    print("-" * 40)
    
    X_test = np.random.randn(10, 20)
    result = dl_toolkit.predict(model_name="stress_predictor", X=X_test)
    
    print(f"Success: {result['success']}")
    print(f"Predictions shape: {np.array(result['predictions']).shape}")
    
    # Example 5: Create a CNN model
    print("\nExample 5: Creating a CNN model")
    print("-" * 40)
    
    result = dl_toolkit.create_cnn_model(
        input_shape=(28, 28, 1),
        output_units=10,
        conv_layers=[32, 64],
        model_name="image_classifier"
    )
    
    print(f"Success: {result['success']}")
    print(f"Model name: {result['model_name']}")
    print(f"Total parameters: {result['total_params']:,}")


if __name__ == '__main__':
    main()

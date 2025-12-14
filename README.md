# StressGuard

An AI-powered chatbot with local LLM integration and ML/DL tools for the Samsung Innovation Campus project.

## 📋 Overview

StressGuard is an Artificial Intelligence project built with Python that combines:
- **Local LLM (Large Language Model)** for natural language understanding and generation
- **Machine Learning tools** for data analysis, classification, regression, and clustering
- **Deep Learning tools** for advanced neural network tasks
- **Interactive chatbot interface** for seamless user interaction

This project demonstrates the integration of various AI technologies to create a comprehensive conversational AI system.

## ✨ Features

### 🤖 Chatbot
- Local LLM integration using HuggingFace Transformers
- Conversation history management
- Context-aware responses
- Interactive CLI interface

### 🔬 Machine Learning Tools
- Classification with Random Forest
- Regression analysis
- Clustering (K-Means)
- Model training and prediction
- Data preprocessing

### 🧠 Deep Learning Tools
- Sequential neural network models
- Convolutional Neural Networks (CNN)
- Custom model architectures
- TensorFlow/Keras integration
- GPU acceleration support

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster inference

### Setup

1. Clone the repository:
```bash
git clone https://github.com/https-ksperez/StressGuard.git
cd StressGuard
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Install the package:
```bash
pip install -e .
```

## 📖 Usage

### Interactive Chatbot

Run the chatbot in interactive mode:
```bash
python src/chatbot/main.py
```

Or if installed:
```bash
stressguard
```

### Command-line Options

```bash
# Use a specific model
python src/chatbot/main.py --model microsoft/DialoGPT-small

# Single message mode
python src/chatbot/main.py --message "Hello, how are you?"

# Use a configuration file
python src/chatbot/main.py --config config/default_config.yaml

# Disable ML or DL tools
python src/chatbot/main.py --no-ml --no-dl
```

### Chatbot Commands

While in interactive mode:
- Type your message to chat with the bot
- `info` - Display chatbot information
- `reset` - Clear conversation history
- `quit` or `exit` - Exit the chatbot

### Python API

#### Basic Chatbot Usage
```python
from chatbot.bot import StressGuardBot

# Initialize chatbot
bot = StressGuardBot(
    model_name="microsoft/DialoGPT-medium",
    enable_ml_tools=True,
    enable_dl_tools=True
)

# Chat with the bot
response = bot.chat("Hello! How can you help me?")
print(response)

# Get chatbot info
info = bot.get_info()
print(info)
```

#### Machine Learning Tools
```python
from ml_tools.ml_toolkit import MLToolkit
import numpy as np

# Initialize ML toolkit
ml = MLToolkit()

# Train a classifier
X = np.random.randn(100, 10)
y = np.random.randint(0, 3, 100)

result = ml.train_classifier(X, y, model_name="my_classifier")
print(f"Test accuracy: {result['test_accuracy']}")

# Make predictions
X_new = np.random.randn(5, 10)
predictions = ml.predict(X_new, model_name="my_classifier")
```

#### Deep Learning Tools
```python
from dl_tools.dl_toolkit import DLToolkit
import numpy as np

# Initialize DL toolkit
dl = DLToolkit()

# Create a neural network
dl.create_sequential_model(
    input_shape=(20,),
    output_units=3,
    hidden_layers=[64, 32],
    model_name="my_model"
)

# Compile the model
dl.compile_model(
    model_name="my_model",
    optimizer="adam",
    loss="sparse_categorical_crossentropy"
)

# Train the model
X_train = np.random.randn(1000, 20)
y_train = np.random.randint(0, 3, 1000)

dl.train_model(
    model_name="my_model",
    X_train=X_train,
    y_train=y_train,
    epochs=10
)
```

## 📁 Project Structure

```
StressGuard/
├── src/
│   ├── chatbot/          # Chatbot core functionality
│   │   ├── __init__.py
│   │   ├── bot.py        # Main chatbot class
│   │   ├── llm_model.py  # LLM model manager
│   │   └── main.py       # CLI entry point
│   ├── ml_tools/         # Machine Learning tools
│   │   ├── __init__.py
│   │   └── ml_toolkit.py
│   ├── dl_tools/         # Deep Learning tools
│   │   ├── __init__.py
│   │   └── dl_toolkit.py
│   └── utils/            # Utility functions
│       ├── __init__.py
│       ├── config.py     # Configuration management
│       └── logger.py     # Logging utilities
├── tests/                # Unit tests
│   ├── conftest.py
│   ├── test_chatbot.py
│   ├── test_ml_toolkit.py
│   ├── test_dl_toolkit.py
│   └── test_utils.py
├── examples/             # Example scripts
│   ├── basic_chatbot.py
│   ├── ml_example.py
│   └── dl_example.py
├── config/               # Configuration files
│   └── default_config.yaml
├── requirements.txt      # Python dependencies
├── setup.py             # Package setup
├── .gitignore
└── README.md
```

## 🧪 Running Tests

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_ml_toolkit.py -v
```

## 📚 Examples

Check the `examples/` directory for comprehensive usage examples:

- `basic_chatbot.py` - Basic chatbot interaction
- `ml_example.py` - Machine learning toolkit examples
- `dl_example.py` - Deep learning toolkit examples

Run an example:
```bash
python examples/ml_example.py
```

## ⚙️ Configuration

Create a custom configuration file (YAML or JSON):

```yaml
# config/my_config.yaml
model:
  name: "microsoft/DialoGPT-medium"
  max_length: 1000
  temperature: 0.7

chatbot:
  enable_ml_tools: true
  enable_dl_tools: true
  max_conversation_history: 20

ml:
  default_test_size: 0.2
  random_state: 42

dl:
  default_epochs: 10
  default_batch_size: 32
```

Use it with:
```bash
python src/chatbot/main.py --config config/my_config.yaml
```

## 🤝 Contributing

This is a Samsung Innovation Campus project. Contributions, issues, and feature requests are welcome!

## 📝 License

This project is created for educational purposes as part of the Samsung Innovation Campus program.

## 👥 Authors

Samsung Innovation Campus Team

## 🙏 Acknowledgments

- Samsung Innovation Campus for the opportunity
- HuggingFace for Transformers library
- TensorFlow and scikit-learn communities

## 📞 Support

For questions or support, please open an issue in the repository. 

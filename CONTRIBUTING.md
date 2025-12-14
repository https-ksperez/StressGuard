# Contributing to StressGuard

Thank you for your interest in contributing to StressGuard! This project is part of the Samsung Innovation Campus program.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/StressGuard.git
   cd StressGuard
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov black flake8
   ```

## Development Workflow

### Using the Dev Script

We provide a helper script for common tasks:

```bash
# Install dependencies
./dev.sh install-dev

# Run tests
./dev.sh test

# Run specific tests
./dev.sh test-ml
./dev.sh test-dl

# Run examples
./dev.sh example-ml
./dev.sh example-dl

# Clean temporary files
./dev.sh clean
```

### Manual Commands

Run tests:
```bash
PYTHONPATH=src pytest tests/
```

Run a specific test file:
```bash
PYTHONPATH=src pytest tests/test_ml_toolkit.py -v
```

Format code with black:
```bash
black src/ tests/ examples/
```

Run linting:
```bash
flake8 src/ tests/ examples/
```

## Code Style

- Follow PEP 8 guidelines
- Use Black for code formatting (line length: 100)
- Add docstrings to all functions and classes
- Write descriptive variable and function names

## Testing

- Write tests for all new features
- Ensure all tests pass before submitting a PR
- Aim for high code coverage
- Test files should be in the `tests/` directory
- Test file names should start with `test_`

## Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Add your descriptive commit message"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request on GitHub

5. Ensure:
   - All tests pass
   - Code is properly formatted
   - Documentation is updated if needed
   - Commit messages are clear and descriptive

## Project Structure

```
StressGuard/
├── src/              # Source code
│   ├── chatbot/      # Chatbot implementation
│   ├── ml_tools/     # Machine learning tools
│   ├── dl_tools/     # Deep learning tools
│   └── utils/        # Utility functions
├── tests/            # Test files
├── examples/         # Example scripts
├── config/           # Configuration files
└── docs/             # Documentation
```

## Adding New Features

### Adding a New ML/DL Tool

1. Implement the tool in the appropriate toolkit (`ml_tools/` or `dl_tools/`)
2. Add comprehensive docstrings
3. Write tests in the `tests/` directory
4. Add an example in the `examples/` directory
5. Update the README if needed

### Adding a New Example

1. Create a new file in `examples/`
2. Add clear comments and documentation
3. Ensure it runs without errors
4. Update the README to mention the new example

## Questions or Issues?

- Open an issue on GitHub
- Contact the Samsung Innovation Campus team

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

Thank you for contributing to StressGuard! 🚀

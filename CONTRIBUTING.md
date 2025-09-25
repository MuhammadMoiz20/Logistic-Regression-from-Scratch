# Contributing to Logistic Regression from Scratch

Thank you for your interest in contributing to this project! This document provides guidelines and information for contributors.

## 🎯 How to Contribute

### Reporting Issues

- Use the GitHub issue tracker to report bugs or request features
- Provide detailed information about the issue, including:
  - Python version
  - Operating system
  - Steps to reproduce
  - Expected vs actual behavior

### Suggesting Enhancements

- Open an issue with the "enhancement" label
- Describe the proposed feature in detail
- Explain why it would be valuable to the project

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**:
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed
4. **Test your changes**: `pytest`
5. **Commit your changes**: `git commit -m "Add your feature"`
6. **Push to your branch**: `git push origin feature/your-feature-name`
7. **Open a Pull Request**

## 📋 Development Setup

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

```bash
# Clone your fork
git clone https://github.com/yourusername/logistic-regression-from-scratch.git
cd logistic-regression-from-scratch

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/logistic_regression_from_scratch --cov-report=html

# Run specific test file
pytest tests/test_core.py -v
```

## 🎨 Coding Standards

### Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [flake8](https://flake8.pycqa.org/) for linting
- Use [mypy](http://mypy-lang.org/) for type checking

### Code Formatting

```bash
# Format code with Black
black src/ tests/

# Check code style with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/
```

### Documentation

- Write clear docstrings for all functions and classes
- Use Google-style docstrings
- Include type hints for all function parameters and return values
- Update README.md for user-facing changes

### Testing

- Write tests for all new functionality
- Aim for high test coverage (>90%)
- Use descriptive test names
- Include both unit tests and integration tests

## 🏗 Project Structure

```
src/logistic_regression_from_scratch/
├── __init__.py              # Package initialization
├── core.py                  # Core algorithms
├── utils.py                 # Utilities and helpers
└── sklearn_integration.py   # Sklearn integration

tests/
├── test_core.py            # Core functionality tests
└── test_utils.py           # Utility function tests

examples/
└── demo.py                 # Example usage

docs/
└── images/                 # Documentation images
```

## 🔍 Review Process

### Pull Request Guidelines

- Keep PRs focused and small when possible
- Include a clear description of changes
- Reference any related issues
- Ensure all tests pass
- Update documentation as needed

### Review Criteria

- Code quality and style
- Test coverage
- Documentation completeness
- Performance considerations
- Backward compatibility

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment Information**:
   - Python version
   - Operating system
   - Package versions

2. **Reproduction Steps**:
   - Minimal code example
   - Expected behavior
   - Actual behavior

3. **Additional Context**:
   - Error messages
   - Screenshots (if applicable)
   - Related issues

## 💡 Feature Requests

When suggesting features:

1. **Problem Description**: What problem does this solve?
2. **Proposed Solution**: How should it work?
3. **Alternatives**: What other approaches were considered?
4. **Additional Context**: Any other relevant information

## 📚 Resources

- [Python Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Autograd Documentation](https://github.com/HIPS/autograd)
- [pytest Documentation](https://docs.pytest.org/)

## 📞 Getting Help

- Open an issue for questions
- Check existing issues and discussions
- Review the documentation

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to this project! 🎉

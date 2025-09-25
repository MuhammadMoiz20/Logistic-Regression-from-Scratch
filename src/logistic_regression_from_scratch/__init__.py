"""
Logistic Regression from Scratch

A comprehensive implementation of logistic regression algorithms including
sigmoid activation, cross-entropy loss, perceptron cost, and gradient descent
optimization. This package demonstrates machine learning fundamentals with
clean, well-documented code suitable for educational and production use.

Key Features:
- Sigmoid activation function implementation
- Cross-entropy and perceptron cost functions
- Gradient descent optimization
- Logistic regression classifier
- Model evaluation with confusion matrix and accuracy metrics
- ROC curve analysis and AUC scoring
- Comprehensive test suite with high coverage

Author: Muhammad Moiz
License: MIT
"""

from .core import (
    sigmoid,
    linear_model,
    cross_entropy_cost,
    perceptron_cost,
    gradient_descent,
    logistic_regression_classifier,
    logistic_regression_binary,
    evaluate_classifier
)

from .utils import (
    generate_synthetic_data,
    plot_results,
    save_results
)

# Optional sklearn integration
try:
    from .sklearn_integration import (
        SklearnLogisticRegression,
        evaluate_with_sklearn
    )
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

__version__ = "1.0.0"
__author__ = "Muhammad Moiz"
__email__ = "moiz@example.com"

__all__ = [
    "sigmoid",
    "linear_model", 
    "cross_entropy_cost",
    "perceptron_cost",
    "gradient_descent",
    "logistic_regression_classifier",
    "logistic_regression_binary",
    "evaluate_classifier",
    "generate_synthetic_data",
    "plot_results",
    "save_results"
]

# Add sklearn functions if available
if _SKLEARN_AVAILABLE:
    __all__.extend([
        "SklearnLogisticRegression",
        "evaluate_with_sklearn"
    ])

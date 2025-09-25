"""
Core logistic regression algorithms and mathematical functions.

This module contains the fundamental implementations of logistic regression
components including activation functions, cost functions, optimization algorithms,
and classification models.
"""

import autograd.numpy as np
from autograd import grad
from typing import Tuple, List, Union


def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute the sigmoid activation function.
    
    The sigmoid function maps any real-valued number to a value between 0 and 1,
    making it ideal for binary classification problems.
    
    Args:
        x: Input value(s) to apply sigmoid function to
        
    Returns:
        Sigmoid activation: 1 / (1 + exp(-x))
        
    Examples:
        >>> sigmoid(0)
        0.5
        >>> sigmoid(-1.5)
        0.18242552380635635
        >>> sigmoid(2.5)
        0.9241418199787566
    """
    return 1 / (1 + np.exp(-x))


def linear_model(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Compute linear model predictions.
    
    Implements the linear model: y = w[0] + w[1] * x
    
    Args:
        x: Input features (1D array)
        w: Weight parameters [bias, slope]
        
    Returns:
        Linear predictions
        
    Examples:
        >>> x = np.array([1, 2, 3])
        >>> w = np.array([0.5, 2.0])
        >>> linear_model(x, w)
        array([2.5, 4.5, 6.5])
    """
    return w[0] + w[1] * x


def cross_entropy_cost(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute cross-entropy cost function for logistic regression.
    
    Cross-entropy is the standard cost function for logistic regression,
    providing smooth gradients and effective penalization of misclassifications.
    
    Args:
        w: Weight parameters [bias, slope]
        x: Input features
        y: Binary target labels (0 or 1)
        
    Returns:
        Cross-entropy cost value
        
    Examples:
        >>> w = np.array([0.0, 1.0])
        >>> x = np.array([1, 2, 3])
        >>> y = np.array([0, 1, 1])
        >>> cost = cross_entropy_cost(w, x, y)
        >>> isinstance(cost, float)
        True
    """
    z = linear_model(x, w)
    pred = sigmoid(z)
    epsilon = 1e-8  # Prevent log(0)
    cost = -np.sum(y * np.log(pred + epsilon) + (1 - y) * np.log(1 - pred + epsilon))
    return cost


def perceptron_cost(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute perceptron cost function (softmax version).
    
    The perceptron cost uses a softplus approximation to the max function,
    providing an alternative to cross-entropy for binary classification.
    
    Args:
        w: Weight parameters [bias, slope]
        x: Input features
        y: Binary target labels (0 or 1)
        
    Returns:
        Perceptron cost value
        
    Examples:
        >>> w = np.array([0.0, 1.0])
        >>> x = np.array([1, 2, 3])
        >>> y = np.array([0, 1, 1])
        >>> cost = perceptron_cost(w, x, y)
        >>> isinstance(cost, float)
        True
    """
    y_modified = 2 * y - 1  # Convert to {-1, 1}
    z = linear_model(x, w)
    # Softplus version: cost = sum(log(1 + exp(-y * z)))
    cost = np.sum(np.log(1 + np.exp(-y_modified * z)))
    return cost


def gradient_descent(
    cost_function, 
    alpha: float, 
    max_its: int, 
    w: List[float], 
    x: np.ndarray, 
    y: np.ndarray
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Perform gradient descent optimization.
    
    Implements gradient descent algorithm to minimize the given cost function
    by iteratively updating weights in the direction of steepest descent.
    
    Args:
        cost_function: Cost function to minimize (e.g., cross_entropy_cost)
        alpha: Learning rate (step size)
        max_its: Maximum number of iterations
        w: Initial weight parameters
        x: Input features
        y: Target labels
        
    Returns:
        Tuple of (weight_history, cost_history)
        
    Examples:
        >>> w_init = [0.0, 1.0]
        >>> x = np.array([1, 2, 3])
        >>> y = np.array([0, 1, 1])
        >>> weights, costs = gradient_descent(cross_entropy_cost, 0.1, 100, w_init, x, y)
        >>> len(weights) == len(costs) == 100
        True
    """
    gradient = grad(cost_function)
    w = np.array(w, dtype=float)
    weight_history = []
    cost_history = []
    
    for k in range(max_its):
        weight_history.append(w.copy())
        cost_history.append(cost_function(w, x, y))
        w -= alpha * gradient(w, x, y)
    
    return weight_history, cost_history


def logistic_regression_classifier(learned_w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Logistic regression classifier that returns probabilities.
    
    Uses learned weights to compute probability of positive class
    for each input sample.
    
    Args:
        learned_w: Learned weight parameters
        x: Input features
        
    Returns:
        Probability predictions for positive class
        
    Examples:
        >>> w = np.array([0.0, 1.0])
        >>> x = np.array([1, 2, 3])
        >>> probs = logistic_regression_classifier(w, x)
        >>> np.all((probs >= 0) & (probs <= 1))
        True
    """
    return sigmoid(linear_model(x, learned_w))


def logistic_regression_binary(
    learned_w: np.ndarray, 
    x: np.ndarray, 
    threshold: float = 0.5
) -> np.ndarray:
    """
    Binary logistic regression classifier with threshold.
    
    Converts probability predictions to binary class labels using
    a specified threshold.
    
    Args:
        learned_w: Learned weight parameters
        x: Input features
        threshold: Classification threshold (default: 0.5)
        
    Returns:
        Binary class predictions (0 or 1)
        
    Examples:
        >>> w = np.array([0.0, 1.0])
        >>> x = np.array([1, 2, 3])
        >>> preds = logistic_regression_binary(w, x, 0.5)
        >>> np.all(np.isin(preds, [0, 1]))
        True
    """
    prob = logistic_regression_classifier(learned_w, x)
    return np.where(prob >= threshold, 1, 0)


def evaluate_classifier(y_actual: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int, float]:
    """
    Evaluate binary classifier performance.
    
    Computes confusion matrix components and accuracy for binary
    classification results.
    
    Args:
        y_actual: True binary labels
        y_pred: Predicted binary labels
        
    Returns:
        Tuple of (false_positive, false_negative, true_positive, true_negative, accuracy)
        
    Examples:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 0])
        >>> fp, fn, tp, tn, acc = evaluate_classifier(y_true, y_pred)
        >>> isinstance(acc, float) and 0 <= acc <= 1
        True
    """
    y_actual = y_actual.flatten()
    y_pred = y_pred.flatten()
    
    true_positive = np.sum((y_actual == 1) & (y_pred == 1))
    true_negative = np.sum((y_actual == 0) & (y_pred == 0))
    false_positive = np.sum((y_actual == 0) & (y_pred == 1))
    false_negative = np.sum((y_actual == 1) & (y_pred == 0))
    
    accuracy = (true_positive + true_negative) / len(y_actual)
    return false_positive, false_negative, true_positive, true_negative, accuracy

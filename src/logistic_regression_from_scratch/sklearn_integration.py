"""
Integration with scikit-learn for advanced evaluation and comparison.

This module provides wrappers and utilities for using scikit-learn's
logistic regression alongside our from-scratch implementation.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class SklearnLogisticRegression:
    """
    Wrapper class for scikit-learn logistic regression with evaluation utilities.
    
    Provides a convenient interface for training sklearn's logistic regression
    and evaluating it with ROC curves and AUC scores.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the sklearn logistic regression wrapper.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.model = LogisticRegression(random_state=random_state)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SklearnLogisticRegression':
        """
        Fit the logistic regression model.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Self for method chaining
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled, y.flatten())
        self.is_fitted = True
        
        return self
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Array of shape (n_samples, 2) with probabilities for each class
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


def evaluate_with_sklearn(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    plot_roc: bool = True
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete evaluation pipeline using scikit-learn logistic regression.
    
    Performs train-test split, fits model, and evaluates with ROC curve analysis.
    
    Args:
        X: Input features
        y: Target labels
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        plot_roc: Whether to plot ROC curve
        
    Returns:
        Tuple of (auc_score, fpr, tpr, thresholds)
        
    Examples:
        >>> X = np.random.randn(100, 5)
        >>> y = np.random.randint(0, 2, (1, 100))
        >>> auc, fpr, tpr, thresh = evaluate_with_sklearn(X, y)
        >>> isinstance(auc, float) and 0 <= auc <= 1
        True
    """
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X.T, y.flatten(), test_size=test_size, random_state=random_state
    )
    
    # Fit model
    model = SklearnLogisticRegression(random_state=random_state)
    model.fit(X_train, y_train)
    
    # Get predictions
    probs = model.predict_proba(X_test)
    probs_class1 = probs[:, 1]
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, probs_class1)
    auc_score = roc_auc_score(y_test, probs_class1)
    
    # Plot ROC curve if requested
    if plot_roc:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, marker='.', linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Logistic Regression')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print(f"AUC Score: {auc_score:.6f}")
        print(f"ROC Thresholds: {thresholds}")
    
    return auc_score, fpr, tpr, thresholds


def compare_implementations(
    X: np.ndarray,
    y: np.ndarray,
    from_scratch_w: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Compare from-scratch implementation with sklearn implementation.
    
    Args:
        X: Input features
        y: Target labels
        from_scratch_w: Learned weights from our implementation
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with comparison results
    """
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X.T, y.flatten(), test_size=test_size, random_state=random_state
    )
    
    # Evaluate sklearn implementation
    sklearn_auc, sklearn_fpr, sklearn_tpr, sklearn_thresh = evaluate_with_sklearn(
        X_train.T, y_train.reshape(1, -1), plot_roc=False
    )
    
    # Evaluate from-scratch implementation
    from .core import logistic_regression_classifier
    from_scratch_probs = logistic_regression_classifier(from_scratch_w, X_test.T)
    from_scratch_fpr, from_scratch_tpr, from_scratch_thresh = roc_curve(
        y_test, from_scratch_probs.flatten()
    )
    from_scratch_auc = roc_auc_score(y_test, from_scratch_probs.flatten())
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(sklearn_fpr, sklearn_tpr, linewidth=2, 
             label=f'Sklearn (AUC = {sklearn_auc:.3f})')
    plt.plot(from_scratch_fpr, from_scratch_tpr, linewidth=2,
             label=f'From Scratch (AUC = {from_scratch_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Implementation Comparison: ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return {
        'sklearn_auc': sklearn_auc,
        'from_scratch_auc': from_scratch_auc,
        'auc_difference': abs(sklearn_auc - from_scratch_auc),
        'sklearn_thresholds': sklearn_thresh,
        'from_scratch_thresholds': from_scratch_thresh
    }

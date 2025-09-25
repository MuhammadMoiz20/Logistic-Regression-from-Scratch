"""
Test suite for logistic regression from scratch implementation.

This module contains comprehensive unit tests for all components
of the logistic regression package.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from logistic_regression_from_scratch.core import (
    sigmoid, linear_model, cross_entropy_cost, perceptron_cost,
    gradient_descent, logistic_regression_classifier,
    logistic_regression_binary, evaluate_classifier
)
from logistic_regression_from_scratch.utils import (
    generate_synthetic_data, generate_heart_disease_data
)


class TestSigmoid:
    """Test sigmoid activation function."""
    
    def test_sigmoid_scalar(self):
        """Test sigmoid with scalar inputs."""
        assert sigmoid(0) == 0.5
        assert sigmoid(-np.inf) == 0.0
        assert sigmoid(np.inf) == 1.0
        
    def test_sigmoid_array(self):
        """Test sigmoid with array inputs."""
        x = np.array([-1, 0, 1])
        result = sigmoid(x)
        expected = np.array([1/(1+np.exp(1)), 0.5, 1/(1+np.exp(-1))])
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_sigmoid_range(self):
        """Test sigmoid output range."""
        x = np.random.uniform(-10, 10, 100)
        result = sigmoid(x)
        assert np.all(result >= 0)
        assert np.all(result <= 1)


class TestLinearModel:
    """Test linear model function."""
    
    def test_linear_model_basic(self):
        """Test basic linear model functionality."""
        x = np.array([1, 2, 3])
        w = np.array([0.5, 2.0])
        result = linear_model(x, w)
        expected = np.array([2.5, 4.5, 6.5])
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_linear_model_zero_weights(self):
        """Test linear model with zero weights."""
        x = np.array([1, 2, 3])
        w = np.array([0.0, 0.0])
        result = linear_model(x, w)
        expected = np.zeros_like(x)
        np.testing.assert_array_almost_equal(result, expected)


class TestCostFunctions:
    """Test cost functions."""
    
    def setup_method(self):
        """Set up test data."""
        self.x = np.array([1, 2, 3])
        self.y = np.array([0, 1, 1])
        self.w = np.array([0.0, 1.0])
        
    def test_cross_entropy_cost(self):
        """Test cross-entropy cost function."""
        cost = cross_entropy_cost(self.w, self.x, self.y)
        assert isinstance(cost, float)
        assert cost >= 0
        
    def test_perceptron_cost(self):
        """Test perceptron cost function."""
        cost = perceptron_cost(self.w, self.x, self.y)
        assert isinstance(cost, float)
        assert cost >= 0
        
    def test_cost_functions_decrease(self):
        """Test that cost functions decrease with better weights."""
        # Perfect weights for this simple case
        w_good = np.array([-1.0, 1.0])
        w_bad = np.array([1.0, -1.0])
        
        cost_good = cross_entropy_cost(w_good, self.x, self.y)
        cost_bad = cross_entropy_cost(w_bad, self.x, self.y)
        
        assert cost_good < cost_bad


class TestGradientDescent:
    """Test gradient descent optimization."""
    
    def setup_method(self):
        """Set up test data."""
        self.x = np.array([1, 2, 3])
        self.y = np.array([0, 1, 1])
        self.w_init = [0.0, 1.0]
        
    def test_gradient_descent_convergence(self):
        """Test gradient descent convergence."""
        weight_history, cost_history = gradient_descent(
            cross_entropy_cost, 0.1, 100, self.w_init, self.x, self.y
        )
        
        assert len(weight_history) == 100
        assert len(cost_history) == 100
        assert cost_history[-1] < cost_history[0]  # Cost should decrease
        
    def test_gradient_descent_parameters(self):
        """Test gradient descent with different parameters."""
        # Test with different learning rates
        for alpha in [0.01, 0.1, 0.5]:
            weight_history, cost_history = gradient_descent(
                cross_entropy_cost, alpha, 50, self.w_init, self.x, self.y
            )
            assert len(weight_history) == 50
            assert len(cost_history) == 50


class TestClassifiers:
    """Test classification functions."""
    
    def setup_method(self):
        """Set up test data."""
        self.x = np.array([1, 2, 3])
        self.w = np.array([0.0, 1.0])
        
    def test_logistic_regression_classifier(self):
        """Test probability classifier."""
        probs = logistic_regression_classifier(self.w, self.x)
        assert len(probs) == len(self.x)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        
    def test_logistic_regression_binary(self):
        """Test binary classifier."""
        preds = logistic_regression_binary(self.w, self.x, threshold=0.5)
        assert len(preds) == len(self.x)
        assert np.all(np.isin(preds, [0, 1]))
        
    def test_binary_classifier_threshold(self):
        """Test binary classifier with different thresholds."""
        probs = logistic_regression_classifier(self.w, self.x)
        
        # Test different thresholds
        for threshold in [0.3, 0.5, 0.7]:
            preds = logistic_regression_binary(self.w, self.x, threshold)
            assert np.all(np.isin(preds, [0, 1]))


class TestEvaluation:
    """Test evaluation functions."""
    
    def test_evaluate_classifier_perfect(self):
        """Test evaluation with perfect predictions."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        
        fp, fn, tp, tn, accuracy = evaluate_classifier(y_true, y_pred)
        
        assert tp == 2
        assert tn == 2
        assert fp == 0
        assert fn == 0
        assert accuracy == 1.0
        
    def test_evaluate_classifier_random(self):
        """Test evaluation with random predictions."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([1, 0, 1, 1])
        
        fp, fn, tp, tn, accuracy = evaluate_classifier(y_true, y_pred)
        
        assert tp == 1
        assert tn == 0
        assert fp == 2
        assert fn == 1
        assert accuracy == 0.25


class TestDataGeneration:
    """Test data generation functions."""
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        x, y = generate_synthetic_data(n_samples=50, random_state=42)
        
        assert x.shape[0] == 1  # Single feature
        assert x.shape[1] == 50  # 50 samples
        assert y.shape[0] == 1  # Single target
        assert y.shape[1] == 50  # 50 samples
        assert np.all(np.isin(y, [0, 1]))  # Binary labels
        
    def test_generate_heart_disease_data(self):
        """Test heart disease data generation."""
        X, y = generate_heart_disease_data(n_samples=100)
        
        assert X.shape[0] > 5  # Multiple features
        assert X.shape[1] == 100  # 100 samples
        assert y.shape[0] == 1  # Single target
        assert y.shape[1] == 100  # 100 samples
        assert np.all(np.isin(y, [0, 1]))  # Binary labels


class TestIntegration:
    """Integration tests for complete workflow."""
    
    def test_complete_workflow(self):
        """Test complete logistic regression workflow."""
        # Generate data
        x, y = generate_synthetic_data(n_samples=20, random_state=42)
        
        # Train model
        w_init = [0.0, 1.0]
        weight_history, cost_history = gradient_descent(
            cross_entropy_cost, 0.1, 100, w_init, x, y
        )
        
        # Make predictions
        learned_w = weight_history[-1]
        predictions = logistic_regression_binary(learned_w, x, threshold=0.5)
        
        # Evaluate
        fp, fn, tp, tn, accuracy = evaluate_classifier(y, predictions)
        
        # Basic assertions
        assert accuracy >= 0
        assert accuracy <= 1
        assert tp + tn + fp + fn == y.size
        
    def test_cost_comparison(self):
        """Test comparison between different cost functions."""
        x, y = generate_synthetic_data(n_samples=20, random_state=42)
        w_init = [0.0, 1.0]
        
        # Train with both cost functions
        _, cost_history_ce = gradient_descent(
            cross_entropy_cost, 0.1, 50, w_init, x, y
        )
        _, cost_history_perc = gradient_descent(
            perceptron_cost, 0.1, 50, w_init, x, y
        )
        
        # Both should converge (cost should decrease)
        assert cost_history_ce[-1] < cost_history_ce[0]
        assert cost_history_perc[-1] < cost_history_perc[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

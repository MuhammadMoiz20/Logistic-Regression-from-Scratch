"""
Utility functions for data generation, visualization, and result management.

This module provides helper functions for creating synthetic datasets,
plotting results, and saving outputs for the logistic regression package.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import os


def generate_synthetic_data(
    n_samples: int = 100,
    n_features: int = 1,
    noise: float = 0.1,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic binary classification dataset.
    
    Creates a linearly separable dataset suitable for logistic regression
    demonstration and testing.
    
    Args:
        n_samples: Number of data points to generate
        n_features: Number of input features (currently only 1D supported)
        noise: Amount of noise to add to the data
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X, y) where X is features and y is binary labels
        
    Examples:
        >>> X, y = generate_synthetic_data(n_samples=50, random_state=42)
        >>> X.shape == (1, 50) and y.shape == (1, 50)
        True
        >>> np.all(np.isin(y, [0, 1]))
        True
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate linearly separable data
    x = np.random.uniform(-3, 3, n_samples)
    # Create a decision boundary at x = 0 with some noise
    y_true = (x > 0).astype(float)
    y = y_true + np.random.normal(0, noise, n_samples)
    y = np.clip(y, 0, 1)  # Ensure binary values
    y = np.round(y).astype(int)
    
    return x.reshape(1, -1), y.reshape(1, -1)


def generate_heart_disease_data(n_samples: int = 304) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic heart disease dataset similar to UCI Heart Disease dataset.
    
    Creates a multi-feature dataset with realistic distributions for
    cardiovascular disease prediction.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (X, y) where X has multiple features and y is binary labels
        
    Examples:
        >>> X, y = generate_heart_disease_data(n_samples=100)
        >>> X.shape[0] > 5 and y.shape[0] == 1
        True
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate features similar to heart disease dataset
    age = np.random.normal(54, 9, n_samples)
    sex = np.random.binomial(1, 0.68, n_samples)  # 68% male
    cp = np.random.choice([0, 1, 2, 3], n_samples, p=[0.46, 0.16, 0.16, 0.22])
    trestbps = np.random.normal(131, 18, n_samples)
    chol = np.random.normal(247, 52, n_samples)
    fbs = np.random.binomial(1, 0.15, n_samples)
    restecg = np.random.choice([0, 1, 2], n_samples, p=[0.48, 0.48, 0.04])
    thalach = np.random.normal(150, 23, n_samples)
    exang = np.random.binomial(1, 0.33, n_samples)
    oldpeak = np.random.exponential(1.04, n_samples)
    slope = np.random.choice([0, 1, 2], n_samples, p=[0.46, 0.48, 0.06])
    ca = np.random.choice([0, 1, 2, 3], n_samples, p=[0.58, 0.24, 0.12, 0.06])
    thal = np.random.choice([0, 1, 2], n_samples, p=[0.31, 0.49, 0.20])
    
    # Stack features
    X = np.column_stack([
        age, sex, cp, trestbps, chol, fbs, restecg, 
        thalach, exang, oldpeak, slope, ca, thal
    ]).T
    
    # Generate target based on some combination of features
    # Higher probability for older males with higher cholesterol
    prob = 1 / (1 + np.exp(-(
        0.02 * age + 
        0.5 * sex + 
        0.3 * cp + 
        0.01 * trestbps + 
        0.005 * chol + 
        0.2 * fbs + 
        0.1 * restecg - 
        0.01 * thalach + 
        0.4 * exang + 
        0.3 * oldpeak + 
        0.2 * slope + 
        0.4 * ca + 
        0.3 * thal - 8
    )))
    
    y = np.random.binomial(1, prob, n_samples).reshape(1, -1)
    
    return X, y


def plot_results(
    x: np.ndarray,
    y: np.ndarray,
    learned_w: np.ndarray,
    cost_history: list,
    title: str = "Logistic Regression Results",
    save_path: Optional[str] = None
) -> None:
    """
    Create comprehensive visualization of logistic regression results.
    
    Generates a multi-panel plot showing data, learned curve, and cost history.
    
    Args:
        x: Input features
        y: Target labels
        learned_w: Learned weight parameters
        cost_history: History of cost values during training
        title: Plot title
        save_path: Optional path to save the plot
        
    Examples:
        >>> x = np.array([[1, 2, 3]])
        >>> y = np.array([[0, 1, 1]])
        >>> w = np.array([0.0, 1.0])
        >>> costs = [1.0, 0.8, 0.6]
        >>> plot_results(x, y, w, costs)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Data and learned curve
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    # Generate smooth curve
    s = np.linspace(np.min(x), np.max(x), 100)
    from .core import logistic_regression_classifier
    predicted_probabilities = logistic_regression_classifier(learned_w, s)
    
    ax1.scatter(x_flat, y_flat, color='blue', alpha=0.7, label='Data Points')
    ax1.plot(s, predicted_probabilities.flatten(), color='red', linewidth=2, 
             label='Learned Logistic Curve')
    ax1.set_xlabel('Input Feature (x)')
    ax1.set_ylabel('Probability / Label')
    ax1.set_title('Logistic Regression Fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cost history
    ax2.plot(cost_history, color='green', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cost')
    ax2.set_title('Training Cost History')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def save_results(
    results: dict,
    filename: str = "results.json"
) -> None:
    """
    Save experiment results to JSON file.
    
    Args:
        results: Dictionary containing results to save
        filename: Output filename
        
    Examples:
        >>> results = {"accuracy": 0.95, "cost": 0.1}
        >>> save_results(results, "test_results.json")
    """
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        else:
            serializable_results[key] = value
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {filename}")


def create_data_files() -> None:
    """
    Create synthetic data files for the examples and tests.
    
    Generates the CSV files referenced in the original notebook.
    """
    # Create 2D classification data
    x, y = generate_synthetic_data(n_samples=11, random_state=42)
    data = np.vstack([x, y])
    np.savetxt('data/2d_classification_data_v1_entropy.csv', data.T, delimiter=',')
    
    # Create heart disease data
    X_heart, y_heart = generate_heart_disease_data(n_samples=304)
    
    # Create header for heart dataset
    header = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    # Combine features and target
    heart_data = np.vstack([X_heart, y_heart]).T
    
    # Save with header
    with open('data/heart.csv', 'w') as f:
        f.write(','.join(header) + '\n')
        np.savetxt(f, heart_data, delimiter=',', fmt='%.6f')
    
    print("Data files created successfully!")

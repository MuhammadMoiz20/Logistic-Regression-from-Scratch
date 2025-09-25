"""
Generate demo images for the README and documentation.

This script creates visualizations to showcase the logistic regression
implementation without requiring sklearn dependencies.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt

# Import only core functions to avoid sklearn dependency
from logistic_regression_from_scratch.core import (
    sigmoid, cross_entropy_cost, perceptron_cost, gradient_descent,
    logistic_regression_classifier, evaluate_classifier
)
from logistic_regression_from_scratch.utils import generate_synthetic_data

# Set style for professional plots
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12


def create_training_convergence_plot():
    """Create training convergence visualization."""
    print("Creating training convergence plot...")
    
    # Generate data
    x, y = generate_synthetic_data(n_samples=100, random_state=42)
    
    # Train with cross-entropy
    w_init = [0.0, 1.0]
    _, cost_history_ce = gradient_descent(
        cross_entropy_cost, 0.1, 1000, w_init, x, y
    )
    
    # Train with perceptron
    _, cost_history_perc = gradient_descent(
        perceptron_cost, 0.1, 1000, w_init, x, y
    )
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Full convergence
    ax1.plot(cost_history_ce, label='Cross-Entropy', color='#2E86AB', linewidth=2)
    ax1.plot(cost_history_perc, label='Perceptron', color='#A23B72', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost')
    ax1.set_title('Training Convergence Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Last 100 iterations detail
    ax2.plot(cost_history_ce[-100:], label='Cross-Entropy (last 100)', color='#2E86AB', linewidth=2)
    ax2.plot(cost_history_perc[-100:], label='Perceptron (last 100)', color='#A23B72', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cost')
    ax2.set_title('Convergence Detail')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('docs/images/training_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Training convergence plot saved")


def create_model_performance_plot():
    """Create model performance visualization."""
    print("Creating model performance plot...")
    
    # Generate data
    x, y = generate_synthetic_data(n_samples=50, random_state=42)
    
    # Train model
    w_init = [0.0, 1.0]
    weight_history, cost_history = gradient_descent(
        cross_entropy_cost, 0.1, 1000, w_init, x, y
    )
    
    learned_w = weight_history[-1]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Data and learned curve
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    # Generate smooth curve
    s = np.linspace(np.min(x), np.max(x), 100)
    predicted_probabilities = logistic_regression_classifier(learned_w, s)
    
    # Plot data points
    colors = ['#FF6B6B' if label == 1 else '#4ECDC4' for label in y_flat]
    ax1.scatter(x_flat, y_flat, c=colors, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    
    # Plot learned curve
    ax1.plot(s, predicted_probabilities.flatten(), color='#2E86AB', linewidth=3, 
             label='Learned Logistic Curve')
    ax1.set_xlabel('Input Feature (x)')
    ax1.set_ylabel('Probability / Label')
    ax1.set_title('Logistic Regression Model Fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cost history
    ax2.plot(cost_history, color='#A23B72', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cost')
    ax2.set_title('Training Cost History')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('docs/images/model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Model performance plot saved")


def create_roc_analysis_plot():
    """Create ROC curve analysis visualization."""
    print("Creating ROC analysis plot...")
    
    # Generate larger dataset for ROC analysis
    x, y = generate_synthetic_data(n_samples=200, random_state=42)
    
    # Train model
    w_init = [0.0, 1.0]
    weight_history, _ = gradient_descent(
        cross_entropy_cost, 0.1, 1000, w_init, x, y
    )
    
    learned_w = weight_history[-1]
    
    # Generate predictions
    probabilities = logistic_regression_classifier(learned_w, x)
    
    # Create ROC curve manually (since we can't use sklearn)
    thresholds = np.linspace(0, 1, 100)
    tpr_values = []
    fpr_values = []
    
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        fp, fn, tp, tn, _ = evaluate_classifier(y, predictions)
        
        if tp + fn > 0:
            tpr = tp / (tp + fn)
        else:
            tpr = 0
            
        if fp + tn > 0:
            fpr = fp / (fp + tn)
        else:
            fpr = 0
            
        tpr_values.append(tpr)
        fpr_values.append(fpr)
    
    # Calculate AUC (approximate)
    auc = np.trapz(tpr_values, fpr_values)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_values, tpr_values, color='#2E86AB', linewidth=3, 
             label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add some styling
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('docs/images/roc_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ ROC analysis plot saved")


def create_sigmoid_demo():
    """Create sigmoid function demonstration."""
    print("Creating sigmoid function demo...")
    
    x = np.linspace(-10, 10, 1000)
    y = sigmoid(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, color='#2E86AB', linewidth=3)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='y = 0.5')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='x = 0')
    plt.xlabel('Input (x)')
    plt.ylabel('Sigmoid Output')
    plt.title('Sigmoid Activation Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('docs/images/sigmoid_function.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Sigmoid function demo saved")


def main():
    """Generate all demo images."""
    print("Generating demo images for documentation...")
    print("=" * 50)
    
    # Create all visualizations
    create_sigmoid_demo()
    create_training_convergence_plot()
    create_model_performance_plot()
    create_roc_analysis_plot()
    
    print("=" * 50)
    print("✓ All demo images generated successfully!")
    print("Images saved to docs/images/")


if __name__ == "__main__":
    main()

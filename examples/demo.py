"""
Demo script showcasing logistic regression from scratch implementation.

This script demonstrates the complete workflow of logistic regression
including data generation, model training, evaluation, and visualization.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from logistic_regression_from_scratch.core import (
    sigmoid, linear_model, cross_entropy_cost, perceptron_cost,
    gradient_descent, logistic_regression_classifier, 
    logistic_regression_binary, evaluate_classifier
)
from logistic_regression_from_scratch.utils import (
    generate_synthetic_data, plot_results, save_results
)


def demo_basic_functions():
    """Demonstrate basic mathematical functions."""
    print("=== Basic Functions Demo ===")
    
    # Test sigmoid function
    test_values = [-1.5, 0, 2.5]
    print("Sigmoid function test:")
    for x in test_values:
        print(f"sigmoid({x}) = {sigmoid(x):.6f}")
    
    print()


def demo_synthetic_data():
    """Demonstrate synthetic data generation and training."""
    print("=== Synthetic Data Training Demo ===")
    
    # Generate synthetic data
    x, y = generate_synthetic_data(n_samples=50, random_state=42)
    print(f"Generated dataset: {x.shape[1]} samples")
    
    # Train with cross-entropy cost
    w_init = [0.0, 1.0]
    alpha = 0.1
    max_its = 1000
    
    print("Training with cross-entropy cost...")
    weight_history, cost_history = gradient_descent(
        cross_entropy_cost, alpha, max_its, w_init, x, y
    )
    
    learned_w = weight_history[-1]
    final_cost = cost_history[-1]
    
    print(f"Final weights: {learned_w}")
    print(f"Final cost: {final_cost:.6f}")
    
    # Evaluate model
    predictions = logistic_regression_binary(learned_w, x, threshold=0.5)
    fp, fn, tp, tn, accuracy = evaluate_classifier(y, predictions)
    
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    
    # Plot results
    plot_results(x, y, learned_w, cost_history, 
                title="Cross-Entropy Cost Training Results")
    
    return learned_w, cost_history


def demo_perceptron_comparison():
    """Compare cross-entropy vs perceptron cost."""
    print("=== Cost Function Comparison Demo ===")
    
    # Generate data
    x, y = generate_synthetic_data(n_samples=50, random_state=42)
    
    # Train with both cost functions
    w_init = [0.0, 1.0]
    alpha = 0.1
    max_its = 1000
    
    print("Training with cross-entropy cost...")
    _, cost_history_ce = gradient_descent(
        cross_entropy_cost, alpha, max_its, w_init, x, y
    )
    
    print("Training with perceptron cost...")
    _, cost_history_perc = gradient_descent(
        perceptron_cost, alpha, max_its, w_init, x, y
    )
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(cost_history_ce, label='Cross-Entropy', color='blue')
    plt.plot(cost_history_perc, label='Perceptron', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Function Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(cost_history_ce[-100:], label='Cross-Entropy (last 100)', color='blue')
    plt.plot(cost_history_perc[-100:], label='Perceptron (last 100)', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Convergence Detail')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Cross-entropy final cost: {cost_history_ce[-1]:.6f}")
    print(f"Perceptron final cost: {cost_history_perc[-1]:.6f}")


def demo_real_data():
    """Demonstrate with real-like heart disease data."""
    print("=== Heart Disease Data Demo ===")
    
    # Load heart disease data
    try:
        data = np.loadtxt('data/heart.csv', delimiter=',', skiprows=1)
        X = data[:, :-1]  # All features except target
        y = data[:, -1].reshape(1, -1)  # Target column
        
        print(f"Heart disease dataset: {X.shape[0]} features, {X.shape[1]} samples")
        
        # Use only first feature for simplicity (age)
        x_simple = X[0:1, :]  # Age feature
        
        # Train model
        w_init = [0.0, 0.1]
        alpha = 0.01
        max_its = 2000
        
        print("Training logistic regression on age vs heart disease...")
        weight_history, cost_history = gradient_descent(
            cross_entropy_cost, alpha, max_its, w_init, x_simple, y
        )
        
        learned_w = weight_history[-1]
        
        # Evaluate
        predictions = logistic_regression_binary(learned_w, x_simple, threshold=0.5)
        fp, fn, tp, tn, accuracy = evaluate_classifier(y, predictions)
        
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        
        # Plot results
        plot_results(x_simple, y, learned_w, cost_history,
                    title="Heart Disease Prediction (Age Feature)")
        
    except FileNotFoundError:
        print("Heart disease data file not found. Skipping real data demo.")


def main():
    """Run complete demo."""
    print("Logistic Regression from Scratch - Complete Demo")
    print("=" * 50)
    
    # Run all demos
    demo_basic_functions()
    learned_w, cost_history = demo_synthetic_data()
    demo_perceptron_comparison()
    demo_real_data()
    
    # Save results
    results = {
        "final_weights": learned_w,
        "final_cost": cost_history[-1],
        "convergence_iterations": len(cost_history)
    }
    save_results(results, "demo_results.json")
    
    print("\nDemo completed successfully!")
    print("Results saved to demo_results.json")


if __name__ == "__main__":
    main()

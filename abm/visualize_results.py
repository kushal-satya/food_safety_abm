#!/usr/bin/env python3
"""
Visualization utilities for the Farmer's Risk Control Behaviors ABM.
This module provides more advanced visualization capabilities for analyzing
simulation results from the model.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns

def ensure_directory(directory):
    """Ensure the output directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_parameter_sensitivity(model, param_name, param_values, output_dir='results'):
    """
    Run the model with different parameter values and plot the results.
    
    Parameters:
    -----------
    model : FarmerRiskControlModel
        The model to run
    param_name : str
        The name of the parameter to vary ('f', 'beta', or 'P')
    param_values : list
        List of values to use for the parameter
    output_dir : str, optional
        Directory to save results
    """
    ensure_directory(output_dir)
    
    # Store results for each parameter value
    mean_alpha_final = []
    mean_contamination_final = []
    mean_cost_final = []
    
    # Original parameters
    original_f = model.f.copy()
    original_beta = model.beta.copy()
    original_P = model.P
    
    # Run simulation for each parameter value
    for value in param_values:
        if param_name == 'f':
            # Multiply all penalties by this value
            model.f = [f * value for f in original_f]
        elif param_name == 'beta':
            # Multiply all testing probabilities by this value
            model.beta = [min(b * value, 1.0) for b in original_beta]
        elif param_name == 'P':
            model.P = value
        else:
            raise ValueError(f"Unknown parameter name: {param_name}")
        
        # Run the simulation
        model.run_simulation()
        
        # Store final values
        mean_alpha_final.append(model.mean_alpha_history[-1])
        mean_contamination_final.append(model.mean_contamination_history[-1])
        mean_cost_final.append(model.mean_cost_history[-1])
    
    # Reset the model parameters
    model.f = original_f
    model.beta = original_beta
    model.P = original_P
    
    # Create plots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    # Parameter labels
    param_labels = {
        'f': 'Penalty Multiplier',
        'beta': 'Testing Probability Multiplier',
        'P': 'Identification Probability'
    }
    
    # Plot alpha vs parameter
    axs[0].plot(param_values, mean_alpha_final, 'o-')
    axs[0].set_xlabel(param_labels[param_name])
    axs[0].set_ylabel('Final Risk Control Effort (α)')
    axs[0].set_title(f'Effect of {param_labels[param_name]} on Risk Control Effort')
    axs[0].grid(True)
    
    # Plot contamination vs parameter
    axs[1].plot(param_values, mean_contamination_final, 'o-')
    axs[1].set_xlabel(param_labels[param_name])
    axs[1].set_ylabel('Final Contamination Rate (σ)')
    axs[1].set_title(f'Effect of {param_labels[param_name]} on Contamination Rate')
    axs[1].grid(True)
    
    # Plot cost vs parameter
    axs[2].plot(param_values, mean_cost_final, 'o-')
    axs[2].set_xlabel(param_labels[param_name])
    axs[2].set_ylabel('Final Cost')
    axs[2].set_title(f'Effect of {param_labels[param_name]} on Cost')
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sensitivity_{param_name}.png'))
    plt.close()

def plot_technology_vs_effort(model, output_dir='results'):
    """
    Create a scatter plot of technology level vs. risk control effort for all farmers.
    
    Parameters:
    -----------
    model : FarmerRiskControlModel
        The model after running a simulation
    output_dir : str, optional
        Directory to save results
    """
    ensure_directory(output_dir)
    
    # Get final values for all farmers
    technology_levels = [farmer.technology_level for farmer in model.farmers]
    alpha_values = [farmer.alpha_history[-1] for farmer in model.farmers]
    contamination_rates = [farmer.contamination_history[-1] for farmer in model.farmers]
    
    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        technology_levels, 
        alpha_values,
        c=contamination_rates,
        cmap='viridis',
        alpha=0.7,
        s=100
    )
    
    plt.colorbar(scatter, label='Contamination Rate (σ)')
    plt.xlabel('Technology Level (k)')
    plt.ylabel('Risk Control Effort (α)')
    plt.title('Technology Level vs. Risk Control Effort')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'technology_vs_effort.png'))
    plt.close()

def plot_neighbor_influence(model, output_dir='results'):
    """
    Visualize the influence of neighbors on farmer's risk control decisions.
    
    Parameters:
    -----------
    model : FarmerRiskControlModel
        The model after running a simulation
    output_dir : str, optional
        Directory to save results
    """
    ensure_directory(output_dir)
    
    # Create a network visualization (simplified)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Position farmers in a circle
    n = model.num_farmers
    radius = 5
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)
    
    # Get final alpha values
    alphas = [farmer.alpha_history[-1] for farmer in model.farmers]
    
    # Normalize for color mapping
    norm = Normalize(vmin=min(alphas), vmax=max(alphas))
    cmap = cm.viridis
    
    # Plot nodes (farmers)
    scatter = ax.scatter(xs, ys, s=300, c=alphas, cmap=cmap, edgecolors='black', linewidths=1)
    
    # Add labels
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.text(x, y, str(i), ha='center', va='center', fontweight='bold')
    
    # Add a colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Risk Control Effort (α)')
    
    ax.set_aspect('equal')
    ax.set_title('Network of Farmers and Their Risk Control Efforts')
    ax.grid(False)
    ax.axis('off')
    
    plt.savefig(os.path.join(output_dir, 'farmer_network.png'))
    plt.close()

def plot_heatmap(model, output_dir='results'):
    """
    Create a heatmap showing the evolution of risk control efforts over time for all farmers.
    
    Parameters:
    -----------
    model : FarmerRiskControlModel
        The model after running a simulation
    output_dir : str, optional
        Directory to save results
    """
    ensure_directory(output_dir)
    
    # Get alpha history for all farmers
    alpha_matrix = np.array([farmer.alpha_history for farmer in model.farmers])
    
    # Create a heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        alpha_matrix,
        cmap='viridis',
        xticklabels=10 if model.time_steps > 10 else 1,
        yticklabels=10 if model.num_farmers > 10 else 1
    )
    plt.xlabel('Time Step')
    plt.ylabel('Farmer ID')
    plt.title('Evolution of Risk Control Efforts Over Time')
    plt.savefig(os.path.join(output_dir, 'effort_heatmap.png'))
    plt.close()

def plot_distributions_over_time(model, time_points, output_dir='results'):
    """
    Plot the distribution of alpha values at different time points.
    
    Parameters:
    -----------
    model : FarmerRiskControlModel
        The model after running a simulation
    time_points : list of int
        List of time points to plot
    output_dir : str, optional
        Directory to save results
    """
    ensure_directory(output_dir)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Get a colormap
    cmap = plt.cm.viridis
    colors = [cmap(i) for i in np.linspace(0, 1, len(time_points))]
    
    # Plot distribution at each time point
    for i, t in enumerate(time_points):
        if t < model.time_steps:
            alpha_values = [farmer.alpha_history[t] for farmer in model.farmers]
            sns.kdeplot(alpha_values, label=f'Time {t}', color=colors[i])
    
    plt.xlabel('Risk Control Effort (α)')
    plt.ylabel('Density')
    plt.title('Distribution of Risk Control Efforts Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'effort_distribution_over_time.png'))
    plt.close()

def analyze_model_results(model, output_dir='results'):
    """
    Run a complete analysis of model results with various visualizations.
    
    Parameters:
    -----------
    model : FarmerRiskControlModel
        The model after running a simulation
    output_dir : str, optional
        Directory to save results
    """
    ensure_directory(output_dir)
    
    # Basic plots (already included in model.plot_results())
    model.plot_results()
    
    # Advanced visualizations
    plot_technology_vs_effort(model, output_dir)
    plot_neighbor_influence(model, output_dir)
    plot_heatmap(model, output_dir)
    
    # Distribution over time
    time_points = [0, model.time_steps // 4, model.time_steps // 2, model.time_steps - 1]
    plot_distributions_over_time(model, time_points, output_dir)
    
    # Parameter sensitivity analysis
    # Only run if not already analyzed in main simulation
    if model.time_steps < 20:  # Only for shorter simulations to save time
        # Penalty multiplier
        penalty_values = [0.5, 1.0, 1.5, 2.0, 2.5]
        plot_parameter_sensitivity(model, 'f', penalty_values, output_dir)
        
        # Testing probability multiplier
        beta_values = [0.5, 1.0, 1.5, 2.0, 2.5]
        plot_parameter_sensitivity(model, 'beta', beta_values, output_dir)
        
        # Identification probability
        P_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        plot_parameter_sensitivity(model, 'P', P_values, output_dir)

if __name__ == "__main__":
    # This module is meant to be imported, but we'll provide a small example
    # if it's run directly
    from farmer_risk_control_model import FarmerRiskControlModel
    
    print("Creating a test model...")
    model = FarmerRiskControlModel(num_farmers=20, time_steps=10, seed=42)
    
    # Set model parameters
    f = [100, 200, 300, 400, 500]
    beta = [0.1, 0.2, 0.3, 0.4]
    P = 0.5
    model.set_parameters(f, beta, P)
    
    print("Running simulation...")
    model.run_simulation()
    
    print("Analyzing results...")
    analyze_model_results(model, 'test_results')
    print("Analysis complete! Results saved in the 'test_results' directory.") 
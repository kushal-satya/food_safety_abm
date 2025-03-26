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
    risk_preferences = [farmer.risk_preference for farmer in model.farmers]
    
    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    
    # Define colors for each risk preference type
    colors = ['blue', 'red', 'green']
    risk_types = sorted(set(risk_preferences))
    
    # Create scatter plot with different colors for risk preferences
    for i, risk_type in enumerate(risk_types):
        indices = [j for j, pref in enumerate(risk_preferences) if pref == risk_type]
        
        plt.scatter(
            [technology_levels[j] for j in indices],
            [alpha_values[j] for j in indices],
            c=[contamination_rates[j] for j in indices],
            cmap='plasma',  # Use a standard colormap that should be available
            alpha=0.7,
            s=100,
            label=f"Risk Type {risk_type}"
        )
    
    plt.colorbar(label='Contamination Rate (σ)')
    plt.xlabel('Technology Level (k)')
    plt.ylabel('Risk Control Effort (α)')
    plt.title('Technology Level vs. Risk Control Effort')
    plt.grid(True)
    plt.legend()
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
    risk_preferences = [farmer.risk_preference for farmer in model.farmers]
    
    # Normalize for color mapping
    norm = Normalize(vmin=min(alphas), vmax=max(alphas))
    
    # Use a colormap that is definitely available
    cmap = plt.cm.plasma
    
    # Create a scatter plot with size based on alpha and color based on risk preference
    scatter = ax.scatter(
        xs, ys, 
        s=[300 * (0.5 + a) for a in alphas],  # Size based on alpha
        c=risk_preferences,  # Color based on risk preference
        cmap=cmap,
        edgecolors='black',
        linewidths=1
    )
    
    # Add labels
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.text(x, y, str(i), ha='center', va='center', fontweight='bold')
    
    # Add a colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Risk Preference Type')
    
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
        cmap='plasma',  # Use a standard colormap
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
    
    # Use a colormap that is definitely available
    cmap = plt.cm.plasma
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

def generate_html_report(output_dir='results'):
    """
    Generate an HTML report with all plots.
    
    Parameters:
    -----------
    output_dir : str, optional
        Directory with the results
    """
    ensure_directory(output_dir)
    
    # Find all PNG files in the output directory
    png_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    png_files.sort()  # Sort by name
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Farmer Risk Control Behaviors - Simulation Results</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
                color: #333;
            }
            h1, h2 {
                color: #0056b3;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
            }
            .plot-container {
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                margin: 20px 0;
                padding: 20px;
            }
            .plot {
                width: 100%;
                max-width: 1000px;
                margin: 10px auto;
                display: block;
            }
            .plot-title {
                font-size: 18px;
                font-weight: bold;
                margin: 10px 0;
                text-align: center;
            }
            .description {
                margin: 15px 0;
                line-height: 1.5;
            }
            .equation {
                font-family: "Times New Roman", serif;
                font-style: italic;
                background-color: #f4f4f4;
                padding: 10px;
                border-radius: 3px;
                text-align: center;
                margin: 10px 0;
            }
            .navbar {
                position: fixed;
                top: 0;
                width: 100%;
                background-color: #343a40;
                padding: 10px 0;
                z-index: 1000;
            }
            .navbar ul {
                list-style-type: none;
                margin: 0;
                padding: 0;
                overflow: hidden;
                display: flex;
                justify-content: center;
            }
            .navbar li {
                margin: 0 15px;
            }
            .navbar a {
                color: white;
                text-decoration: none;
                padding: 8px 12px;
                border-radius: 3px;
                transition: background-color 0.3s;
            }
            .navbar a:hover {
                background-color: #495057;
            }
            .content {
                margin-top: 60px;
            }
        </style>
    </head>
    <body>
        <div class="navbar">
            <ul>
                <li><a href="#overview">Overview</a></li>
                <li><a href="#model">Model Details</a></li>
                <li><a href="#results">Simulation Results</a></li>
                <li><a href="#individual">Individual Farmers</a></li>
            </ul>
        </div>
        
        <div class="content">
            <h1 id="overview">Farmer Risk Control Behaviors - Simulation Results</h1>
            
            <div class="plot-container">
                <p class="description">
                    This report presents the results of an agent-based simulation of farmers' risk control behaviors
                    in a food safety context. The model explores how different types of farmers (risk neutral, risk averse,
                    and risk loving) adjust their risk control efforts in response to testing regimes, penalties, and technology levels.
                </p>
            </div>
            
            <h2 id="model">Model Details</h2>
            
            <div class="plot-container">
                <p class="description">
                    <strong>Key Features of the Model:</strong>
                </p>
                <ul>
                    <li>Three farmer types: Risk Neutral, Risk Averse, and Risk Loving</li>
                    <li>Exponential function for risk probability</li>
                    <li>Cost function based on testing probabilities and penalties</li>
                    <li>Dynamic technology adaptation</li>
                    <li>Neighbor influence on decision making</li>
                </ul>
                
                <p class="description">
                    <strong>Key Equations:</strong>
                </p>
                <p class="description">
                    Contamination Rate Equation (3):
                </p>
                <p class="equation">
                    σ<sub>j</sub><sup>t</sup> = e<sup>(-c<sub>j</sub><sup>t</sup> × k<sub>j</sub>)</sup>
                </p>
                <p class="description">
                    where σ<sub>j</sub><sup>t</sup> is the contamination rate, c<sub>j</sub><sup>t</sup> is the risk control effort,
                    and k<sub>j</sub> is the technology level of farmer j at time t.
                </p>
                
                <p class="description">
                    Cost Function Equation (4):
                </p>
                <p class="equation">
                    f = (f<sub>1</sub>β<sub>1</sub> + f<sub>2</sub>(1-β<sub>1</sub>)β<sub>2</sub> + f<sub>3</sub>(1-β<sub>1</sub>)(1-β<sub>2</sub>)β<sub>3</sub> + f<sub>4</sub>(1-β<sub>1</sub>)(1-β<sub>2</sub>)(1-β<sub>3</sub>)β<sub>4</sub> + f<sub>5</sub>(1-β<sub>1</sub>)(1-β<sub>2</sub>)(1-β<sub>3</sub>)(1-β<sub>4</sub>)P) / ((c<sub>e</sub>+c<sub>k</sub>)(1-β<sub>1</sub>)(1-β<sub>2</sub>)(1-β<sub>3</sub>)(1-β<sub>4</sub>))
                </p>
                <p class="description">
                    where f<sub>1</sub> to f<sub>5</sub> are penalties, β<sub>1</sub> to β<sub>4</sub> are testing probabilities,
                    P is the probability of product identification, and c<sub>e</sub> and c<sub>k</sub> are effort and technology costs.
                </p>
                
                <p class="description">
                    A farmer's optimal contamination rate can be found by minimizing the cost function (Eq. 5).
                </p>
            </div>
            
            <h2 id="results">Simulation Results</h2>
    """
    
    # Add plots to the HTML
    for png_file in png_files:
        plot_name = png_file.replace('.png', '').replace('_', ' ').title()
        
        html_content += f"""
            <div class="plot-container" id="{png_file.replace('.png', '')}">
                <p class="plot-title">{plot_name}</p>
                <img class="plot" src="{png_file}" alt="{plot_name}">
            </div>
        """
    
    html_content += """
            <h2 id="individual">Conclusions</h2>
            <div class="plot-container">
                <p class="description">
                    The simulation results demonstrate how farmers with different risk preferences adjust their
                    risk control efforts over time. Risk averse farmers tend to invest more in risk control measures
                    and technology, resulting in lower contamination rates but higher costs. Risk loving farmers
                    take more chances with lower control efforts, which can result in higher contamination rates
                    but potentially lower costs in some scenarios.
                </p>
                <p class="description">
                    The model illustrates the importance of considering heterogeneous risk preferences in
                    food safety regulation and policy design. Effective policies should consider the diverse
                    behaviors of different farmer types to achieve optimal food safety outcomes.
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write the HTML file
    with open(os.path.join(output_dir, 'report.html'), 'w') as f:
        f.write(html_content)
    
    print(f"HTML report generated at {os.path.join(output_dir, 'report.html')}")

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
    model.plot_results(output_dir)
    
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
    
    # Generate HTML report
    generate_html_report(output_dir)

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
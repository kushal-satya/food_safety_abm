#!/usr/bin/env python3
"""
Extended analysis script for the Farmer's Risk Control Behaviors Model
Runs a longer simulation and generates detailed cost breakdowns and analytical comparisons
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from abm.farmer_risk_control_model import FarmerRiskControlModel, Farmer
from abm.visualize_equations import ensure_directory
from scipy.optimize import minimize

# Constants for longer simulation
NUM_FARMERS = 150
TIME_STEPS = 300  # Much longer time period
SEED = 42

def ensure_directory(directory):
    """Ensure a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def analyze_costs_by_risk_type(model, output_dir):
    """
    Analyze and plot detailed cost breakdown by risk type over time
    """
    # Create directories for output
    ensure_directory(output_dir)
    
    # Track costs by risk type over time
    time_steps = model.time_steps
    time_range = range(time_steps)
    
    # Prepare data structures for costs
    cost_types = ['penalty', 'testing', 'effort', 'technology', 'total']
    risk_types = [Farmer.RISK_NEUTRAL, Farmer.RISK_AVERSE, Farmer.RISK_LOVING]
    risk_type_names = {
        Farmer.RISK_NEUTRAL: "Risk Neutral",
        Farmer.RISK_AVERSE: "Risk Averse",
        Farmer.RISK_LOVING: "Risk Loving"
    }
    
    # Initialize cost arrays
    costs_by_risk_type = {
        risk_type: {
            cost_type: np.zeros(time_steps) for cost_type in cost_types
        } for risk_type in risk_types
    }
    
    # Calculate costs for each time step and risk type
    for t in range(time_steps):
        for risk_type in risk_types:
            farmers_of_type = [f for f in model.farmers if f.risk_preference == risk_type]
            if not farmers_of_type:
                continue
                
            # Calculate average costs for this risk type at this time step
            effort_costs = np.mean([f.c_e * f.alpha_history[t] for f in farmers_of_type])
            tech_costs = np.mean([f.c_k * f.technology_history[t] for f in farmers_of_type])
            
            # Calculate expected penalty cost (approximation)
            avg_contamination = np.mean([f.contamination_history[t] for f in farmers_of_type])
            avg_penalty = avg_contamination * sum(model.f) * 0.1  # Rough estimate
            
            # Testing costs (from model since stored at model level)
            if hasattr(model, 'testing_cost_history_by_risk') and t < len(model.testing_cost_history_by_risk[risk_type]):
                testing_cost = model.testing_cost_history_by_risk[risk_type][t] / len(farmers_of_type)
            else:
                testing_cost = 0
                
            # Total cost
            total_cost = effort_costs + tech_costs + avg_penalty + testing_cost
            
            # Store in our data structure
            costs_by_risk_type[risk_type]['effort'][t] = effort_costs
            costs_by_risk_type[risk_type]['technology'][t] = tech_costs
            costs_by_risk_type[risk_type]['penalty'][t] = avg_penalty
            costs_by_risk_type[risk_type]['testing'][t] = testing_cost
            costs_by_risk_type[risk_type]['total'][t] = total_cost
    
    # Create plots for each cost type
    for cost_type in cost_types:
        plt.figure(figsize=(12, 8))
        for risk_type in risk_types:
            plt.plot(time_range, costs_by_risk_type[risk_type][cost_type], 
                    label=f"{risk_type_names[risk_type]}")
        
        plt.xlabel('Time Step')
        plt.ylabel(f'{cost_type.capitalize()} Cost')
        plt.title(f'Average {cost_type.capitalize()} Cost by Risk Type Over Time')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{cost_type}_cost_by_risk_type.png'))
        plt.close()
    
    # Create stacked bar charts showing cost composition at specific time points
    time_points = [0, int(time_steps/4), int(time_steps/2), int(3*time_steps/4), time_steps-1]
    time_labels = ['Initial', '25%', '50%', '75%', 'Final']
    
    for i, (t, label) in enumerate(zip(time_points, time_labels)):
        plt.figure(figsize=(12, 8))
        bar_width = 0.2
        x = np.arange(len(risk_types))
        
        # Prepare data for stacked bars
        effort = [costs_by_risk_type[rt]['effort'][t] for rt in risk_types]
        tech = [costs_by_risk_type[rt]['technology'][t] for rt in risk_types]
        penalty = [costs_by_risk_type[rt]['penalty'][t] for rt in risk_types]
        testing = [costs_by_risk_type[rt]['testing'][t] for rt in risk_types]
        
        # Create stacked bars
        plt.bar(x, effort, bar_width, label='Effort Cost')
        plt.bar(x, tech, bar_width, bottom=effort, label='Technology Cost')
        plt.bar(x, penalty, bar_width, bottom=np.array(effort) + np.array(tech), label='Penalty Cost')
        plt.bar(x, testing, bar_width, bottom=np.array(effort) + np.array(tech) + np.array(penalty), label='Testing Cost')
        
        plt.xlabel('Risk Type')
        plt.ylabel('Cost')
        plt.title(f'Cost Composition by Risk Type ({label} - Time Step {t})')
        plt.xticks(x, [risk_type_names[rt] for rt in risk_types])
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'cost_composition_time_{t}.png'))
        plt.close()
    
    return costs_by_risk_type

def calculate_analytical_solution(model, time_point):
    """
    Calculate the analytical optimal solution at a specific time point
    and compare with the ABM results
    """
    # Get model parameters at the time point
    f = model.f
    beta = model.beta
    P = model.P
    c_e_avg = np.mean([farmer.c_e for farmer in model.farmers])
    c_k_avg = np.mean([farmer.c_k for farmer in model.farmers])
    
    # Technology levels at this time point by risk type
    technology_levels = {
        Farmer.RISK_NEUTRAL: np.mean([farmer.technology_history[time_point] 
                                      for farmer in model.farmers 
                                      if farmer.risk_preference == Farmer.RISK_NEUTRAL]),
        Farmer.RISK_AVERSE: np.mean([farmer.technology_history[time_point] 
                                    for farmer in model.farmers 
                                    if farmer.risk_preference == Farmer.RISK_AVERSE]),
        Farmer.RISK_LOVING: np.mean([farmer.technology_history[time_point] 
                                    for farmer in model.farmers 
                                    if farmer.risk_preference == Farmer.RISK_LOVING])
    }
    
    # Find analytical optimal effort for each risk type
    analytical_results = {}
    
    # Function to calculate cost for a given alpha
    def cost_function(alpha, tech_level, risk_type):
        # Create a temporary farmer to use its cost calculation method
        temp_farmer = Farmer(0, 1, alpha=alpha[0], technology_level=tech_level, 
                             risk_preference=risk_type)
        
        # Calculate contamination rate
        contamination = temp_farmer.calculate_contamination_rate(0)
        
        # Calculate cost
        cost = temp_farmer.calculate_cost(f, beta, P, c_e_avg, c_k_avg)
        
        return cost if not np.isnan(cost) and not np.isinf(cost) else 1e10
    
    # Find analytical optimum for each risk type
    for risk_type in [Farmer.RISK_NEUTRAL, Farmer.RISK_AVERSE, Farmer.RISK_LOVING]:
        tech_level = technology_levels[risk_type]
        
        # Define bounds for optimization
        bounds = [(0.05, 0.95)]
        
        # Initial guess
        x0 = [0.5]
        
        # Find optimal alpha for this risk type
        result = minimize(lambda x: cost_function(x, tech_level, risk_type), 
                          x0, bounds=bounds, method='SLSQP')
        
        if result.success:
            optimal_alpha = result.x[0]
        else:
            # If optimization fails, do a grid search
            test_alphas = np.linspace(0.05, 0.95, 50)
            costs = [cost_function([alpha], tech_level, risk_type) for alpha in test_alphas]
            optimal_alpha = test_alphas[np.argmin(costs)]
        
        # Calculate corresponding contamination and cost
        temp_farmer = Farmer(0, 1, alpha=optimal_alpha, technology_level=float(tech_level),
                            risk_preference=risk_type)
        optimal_contamination = temp_farmer.calculate_contamination_rate(0)
        optimal_cost = temp_farmer.calculate_cost(f, beta, P, c_e_avg, c_k_avg)
        
        analytical_results[risk_type] = {
            'alpha': optimal_alpha,
            'contamination': optimal_contamination,
            'cost': optimal_cost,
            'technology': tech_level
        }
    
    return analytical_results

def compare_analytical_vs_abm(model, time_point, output_dir):
    """
    Compare analytical optimal solutions with ABM results at a specific time point
    """
    ensure_directory(output_dir)
    
    # Get analytical solutions
    analytical_results = calculate_analytical_solution(model, time_point)
    
    # Get ABM results at time point
    abm_results = {}
    for risk_type in [Farmer.RISK_NEUTRAL, Farmer.RISK_AVERSE, Farmer.RISK_LOVING]:
        farmers_of_type = [f for f in model.farmers if f.risk_preference == risk_type]
        
        if farmers_of_type:
            abm_results[risk_type] = {
                'alpha': np.mean([f.alpha_history[time_point] for f in farmers_of_type]),
                'contamination': np.mean([f.contamination_history[time_point] for f in farmers_of_type]),
                'cost': np.mean([f.cost_history[time_point] for f in farmers_of_type 
                               if not np.isnan(f.cost_history[time_point]) and 
                               not np.isinf(f.cost_history[time_point])]),
                'technology': np.mean([f.technology_history[time_point] for f in farmers_of_type])
            }
    
    # Create comparison plots
    risk_type_names = {
        Farmer.RISK_NEUTRAL: "Risk Neutral",
        Farmer.RISK_AVERSE: "Risk Averse",
        Farmer.RISK_LOVING: "Risk Loving"
    }
    
    # 1. Compare alpha (effort)
    plt.figure(figsize=(10, 6))
    x = np.arange(len(risk_type_names))
    width = 0.35
    
    analytical_alphas = [analytical_results[rt]['alpha'] for rt in risk_type_names.keys()]
    abm_alphas = [abm_results[rt]['alpha'] for rt in risk_type_names.keys()]
    
    plt.bar(x - width/2, analytical_alphas, width, label='Analytical Solution')
    plt.bar(x + width/2, abm_alphas, width, label='ABM Result')
    
    plt.xlabel('Risk Type')
    plt.ylabel('Risk Control Effort (α)')
    plt.title(f'Analytical vs. ABM Risk Control Effort (Time Step {time_point})')
    plt.xticks(x, [name for name in risk_type_names.values()])
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(output_dir, f'analytical_vs_abm_alpha_t{time_point}.png'))
    plt.close()
    
    # 2. Compare contamination rates
    plt.figure(figsize=(10, 6))
    
    analytical_cont = [analytical_results[rt]['contamination'] for rt in risk_type_names.keys()]
    abm_cont = [abm_results[rt]['contamination'] for rt in risk_type_names.keys()]
    
    plt.bar(x - width/2, analytical_cont, width, label='Analytical Solution')
    plt.bar(x + width/2, abm_cont, width, label='ABM Result')
    
    plt.xlabel('Risk Type')
    plt.ylabel('Contamination Rate (σ)')
    plt.title(f'Analytical vs. ABM Contamination Rate (Time Step {time_point})')
    plt.xticks(x, [name for name in risk_type_names.values()])
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(output_dir, f'analytical_vs_abm_contamination_t{time_point}.png'))
    plt.close()
    
    # 3. Compare costs
    plt.figure(figsize=(10, 6))
    
    analytical_costs = [analytical_results[rt]['cost'] for rt in risk_type_names.keys()]
    abm_costs = [abm_results[rt]['cost'] for rt in risk_type_names.keys()]
    
    plt.bar(x - width/2, analytical_costs, width, label='Analytical Solution')
    plt.bar(x + width/2, abm_costs, width, label='ABM Result')
    
    plt.xlabel('Risk Type')
    plt.ylabel('Total Cost')
    plt.title(f'Analytical vs. ABM Total Cost (Time Step {time_point})')
    plt.xticks(x, [name for name in risk_type_names.values()])
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(output_dir, f'analytical_vs_abm_cost_t{time_point}.png'))
    plt.close()
    
    return analytical_results, abm_results

def generate_html_report(model, costs_by_risk_type, analytical_comparison, output_dir):
    """
    Generate an HTML report with all the analysis results
    """
    html_path = os.path.join(output_dir, 'extended_analysis_report.html')
    
    # Get metadata
    time_steps = model.time_steps
    num_farmers = len(model.farmers)
    num_risk_neutral = model.num_risk_neutral
    num_risk_averse = model.num_risk_averse  
    num_risk_loving = model.num_risk_loving
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Extended Analysis of Farmer Risk Control Behaviors</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
                color: #333;
                line-height: 1.6;
            }}
            h1, h2, h3 {{
                color: #0056b3;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                border-radius: 5px;
            }}
            .section {{
                margin-bottom: 40px;
            }}
            .plot {{
                width: 100%;
                max-width: 800px;
                margin: 20px auto;
                display: block;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .plot-container {{
                margin: 20px 0;
            }}
            .metadata {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .metadata-item {{
                margin-bottom: 5px;
            }}
            .multi-plot {{
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 20px;
                margin: 20px 0;
            }}
            .multi-plot img {{
                width: 45%;
                min-width: 400px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .cost-breakdown {{
                margin-top: 30px;
            }}
            .time-points {{
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 20px;
                margin: 20px 0;
            }}
            .time-point {{
                width: 30%;
                min-width: 300px;
                margin-bottom: 20px;
            }}
            .description {{
                margin: 20px 0;
                text-align: justify;
            }}
            .analytical-comparison {{
                margin-top: 30px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Extended Analysis of Farmer Risk Control Behaviors</h1>
            
            <div class="metadata">
                <h3>Simulation Parameters</h3>
                <div class="metadata-item"><strong>Time Steps:</strong> {time_steps}</div>
                <div class="metadata-item"><strong>Number of Farmers:</strong> {num_farmers}</div>
                <div class="metadata-item"><strong>Risk Neutral Farmers:</strong> {num_risk_neutral} ({num_risk_neutral/num_farmers:.1%})</div>
                <div class="metadata-item"><strong>Risk Averse Farmers:</strong> {num_risk_averse} ({num_risk_averse/num_farmers:.1%})</div>
                <div class="metadata-item"><strong>Risk Loving Farmers:</strong> {num_risk_loving} ({num_risk_loving/num_farmers:.1%})</div>
            </div>
            
            <div class="section">
                <h2>Long-Term Behavior by Risk Type</h2>
                <p class="description">
                    This extended analysis examines how farmers with different risk preferences (risk neutral,
                    risk averse, and risk loving) behave over a longer time period of {time_steps} time steps.
                    The plots below show their risk control effort, contamination rates, costs, and technology levels.
                </p>
                
                <div class="plot-container">
                    <h3>Risk Control Effort and Contamination</h3>
                    <img src="simulation_results.png" alt="Simulation Results" class="plot">
                </div>
                
                <div class="plot-container">
                    <h3>Technology Level Over Time</h3>
                    <img src="technology_over_time.png" alt="Technology Level Over Time" class="plot">
                </div>
            </div>
            
            <div class="section cost-breakdown">
                <h2>Cost Breakdown Analysis</h2>
                <p class="description">
                    The following plots break down the different cost components for each risk type.
                    This helps us understand how effort costs, technology costs, penalty costs, and testing costs
                    contribute to the total cost, and how these differ between risk neutral, risk averse,
                    and risk loving farmers.
                </p>
                
                <h3>Cost Components Over Time</h3>
                <div class="multi-plot">
                    <img src="costs/effort_cost_by_risk_type.png" alt="Effort Cost by Risk Type">
                    <img src="costs/technology_cost_by_risk_type.png" alt="Technology Cost by Risk Type">
                    <img src="costs/penalty_cost_by_risk_type.png" alt="Penalty Cost by Risk Type">
                    <img src="costs/testing_cost_by_risk_type.png" alt="Testing Cost by Risk Type">
                </div>
                
                <h3>Total Cost Over Time</h3>
                <img src="costs/total_cost_by_risk_type.png" alt="Total Cost by Risk Type" class="plot">
                
                <h3>Cost Composition at Different Time Points</h3>
                <p class="description">
                    These charts show how the cost components stack up at different points in time,
                    allowing us to see how the cost structure evolves throughout the simulation.
                </p>
                
                <div class="time-points">
                    <div class="time-point">
                        <img src="costs/cost_composition_time_0.png" alt="Cost Composition (Initial)" class="plot">
                    </div>
                    <div class="time-point">
                        <img src="costs/cost_composition_time_{int(time_steps/4)}.png" alt="Cost Composition (25%)" class="plot">
                    </div>
                    <div class="time-point">
                        <img src="costs/cost_composition_time_{int(time_steps/2)}.png" alt="Cost Composition (50%)" class="plot">
                    </div>
                    <div class="time-point">
                        <img src="costs/cost_composition_time_{int(3*time_steps/4)}.png" alt="Cost Composition (75%)" class="plot">
                    </div>
                    <div class="time-point">
                        <img src="costs/cost_composition_time_{time_steps-1}.png" alt="Cost Composition (Final)" class="plot">
                    </div>
                </div>
            </div>
            
            <div class="section analytical-comparison">
                <h2>Analytical Solution vs. ABM Results</h2>
                <p class="description">
                    This section compares the analytical optimization solution with the emergent behavior
                    from the agent-based model at time step {int(time_steps/2)}. The analytical solution
                    represents what would be mathematically optimal based on the cost functions,
                    while the ABM results show how agents actually behave with bounded rationality,
                    path dependence, and inter-agent effects.
                </p>
                
                <div class="multi-plot">
                    <img src="analytical/analytical_vs_abm_alpha_t{int(time_steps/2)}.png" alt="Analytical vs ABM Effort">
                    <img src="analytical/analytical_vs_abm_contamination_t{int(time_steps/2)}.png" alt="Analytical vs ABM Contamination">
                </div>
                
                <img src="analytical/analytical_vs_abm_cost_t{int(time_steps/2)}.png" alt="Analytical vs ABM Cost" class="plot">
                
                <p class="description">
                    The differences between the analytical and ABM results highlight the importance of
                    agent-based modeling for understanding complex systems. While the analytical solution
                    represents the theoretical optimum, the ABM captures how behavioral factors, learning,
                    and social influences affect decision-making over time.
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write the HTML file
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"Extended analysis report generated at {html_path}")
    return html_path

def main():
    """Run extended analysis of the model"""
    # Create output directory
    output_dir = 'results/extended_analysis'
    ensure_directory(output_dir)
    
    print(f"Running extended analysis with {NUM_FARMERS} farmers for {TIME_STEPS} time steps")
    
    # Create and run model
    model = FarmerRiskControlModel(
        num_farmers=NUM_FARMERS,
        time_steps=TIME_STEPS,
        seed=SEED,
        risk_neutral_pct=0.33,
        risk_averse_pct=0.33,
        risk_loving_pct=0.34,
        penalty_multiplier=1.0,
        testing_multiplier=1.0,
        id_probability=0.5
    )
    
    print(f"Created {model.num_risk_neutral} risk neutral farmers")
    print(f"Created {model.num_risk_averse} risk averse farmers")
    print(f"Created {model.num_risk_loving} risk loving farmers")
    
    # Run the simulation
    print("Running long-term simulation...")
    model.run_simulation()
    
    # Save basic results
    model.save_results_to_file(output_dir)
    model.plot_results(output_dir)
    
    # Run extended analysis
    print("Analyzing cost breakdown by risk type...")
    costs_by_risk_type = analyze_costs_by_risk_type(model, os.path.join(output_dir, 'costs'))
    
    # Compare with analytical solution at midpoint of simulation
    midpoint = int(TIME_STEPS / 2)
    print(f"Comparing ABM results with analytical solution at time step {midpoint}...")
    analytical_comparison = compare_analytical_vs_abm(
        model, midpoint, os.path.join(output_dir, 'analytical')
    )
    
    # Generate HTML report
    print("Generating comprehensive HTML report...")
    html_path = generate_html_report(model, costs_by_risk_type, analytical_comparison, output_dir)
    
    print("Extended analysis complete!")
    print(f"Results saved in the '{output_dir}' directory")
    print(f"View the report in your browser by opening: file://{os.path.abspath(html_path)}")

if __name__ == "__main__":
    main() 
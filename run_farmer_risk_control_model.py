#!/usr/bin/env python3
"""
Run script for the Farmer's Risk Control Behaviors Agent-Based Model.
This script sets up and runs the simulation described in Model 3 of
the food safety risk control behaviors study.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from abm.farmer_risk_control_model import FarmerRiskControlModel
from abm.visualize_results import analyze_model_results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run the Farmer Risk Control ABM simulation'
    )
    parser.add_argument(
        '--num_farmers', type=int, default=100,
        help='Number of farmer agents to simulate (default: 100)'
    )
    parser.add_argument(
        '--time_steps', type=int, default=50,
        help='Number of time steps to simulate (default: 50)'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility (default: None)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='results',
        help='Directory to save results (default: results)'
    )
    parser.add_argument(
        '--analysis', action='store_true',
        help='Run detailed analysis and visualization after simulation'
    )
    parser.add_argument(
        '--penalty_multiplier', type=float, default=1.0,
        help='Multiplier for penalty values (default: 1.0)'
    )
    parser.add_argument(
        '--testing_multiplier', type=float, default=1.0,
        help='Multiplier for testing probabilities (default: 1.0)'
    )
    parser.add_argument(
        '--identification_prob', type=float, default=0.5,
        help='Probability of identifying eligible products (default: 0.5)'
    )
    return parser.parse_args()

def setup_model_parameters(args):
    """
    Set up model parameters based on the equations described in the model specification.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
        
    Returns:
    --------
    tuple: f, beta, P values for the model
    """
    # Base penalties for contamination detected at test points 1-4
    # and probability that the farmer's eligible products can be identified through tracing
    f1 = 100 * args.penalty_multiplier  # Penalty at test point 1
    f2 = 200 * args.penalty_multiplier  # Penalty at test point 2 
    f3 = 300 * args.penalty_multiplier  # Penalty at test point 3
    f4 = 400 * args.penalty_multiplier  # Penalty at test point 4
    f5 = 500 * args.penalty_multiplier  # Penalty from resulting illness
    
    # Testing probabilities at each test point
    beta1 = min(0.1 * args.testing_multiplier, 1.0)  # Testing probability at point 1
    beta2 = min(0.2 * args.testing_multiplier, 1.0)  # Testing probability at point 2
    beta3 = min(0.3 * args.testing_multiplier, 1.0)  # Testing probability at point 3
    beta4 = min(0.4 * args.testing_multiplier, 1.0)  # Testing probability at point 4
    
    # Probability that the farmer's eligible products can be identified
    P = args.identification_prob
    
    return [f1, f2, f3, f4, f5], [beta1, beta2, beta3, beta4], P

def ensure_directory(directory):
    """Ensure the output directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    """Main function to run the simulation."""
    args = parse_args()
    ensure_directory(args.output_dir)
    
    print(f"Starting Farmer's Risk Control Behaviors Simulation with {args.num_farmers} farmers")
    print(f"Running for {args.time_steps} time steps")
    print(f"Penalty multiplier: {args.penalty_multiplier}")
    print(f"Testing probability multiplier: {args.testing_multiplier}")
    print(f"Identification probability: {args.identification_prob}")
    
    # Create the model
    model = FarmerRiskControlModel(
        num_farmers=args.num_farmers,
        time_steps=args.time_steps,
        seed=args.seed
    )
    
    # Set model parameters
    f, beta, P = setup_model_parameters(args)
    model.set_parameters(f, beta, P)
    
    # Run the simulation
    print("Running simulation...")
    model.run_simulation()
    
    # Save model parameters to a file
    params_file = os.path.join(args.output_dir, 'parameters.txt')
    with open(params_file, 'w') as f:
        f.write(f"Number of farmers: {args.num_farmers}\n")
        f.write(f"Time steps: {args.time_steps}\n")
        f.write(f"Random seed: {args.seed}\n")
        f.write(f"Penalty values: {model.f}\n")
        f.write(f"Testing probabilities: {model.beta}\n")
        f.write(f"Identification probability: {model.P}\n")
        f.write(f"Effort cost range: {model.c_e_range}\n")
        f.write(f"Technology cost range: {model.c_k_range}\n")
    
    # Plot and save results
    if args.analysis:
        print("Running detailed analysis and visualization...")
        analyze_model_results(model, args.output_dir)
    else:
        print("Plotting basic results...")
        model.plot_results()
    
    print("Simulation complete!")
    print(f"Results saved in the '{args.output_dir}' directory")

if __name__ == "__main__":
    main() 
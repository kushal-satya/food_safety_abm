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
from abm.visualize_results import analyze_model_results, generate_html_report
from abm.visualize_equations import generate_equation_html_report

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
        '--analyze', action='store_true',
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
    parser.add_argument(
        '--risk_neutral_pct', type=float, default=0.33,
        help='Percentage of farmers that are risk neutral (default: 0.33)'
    )
    parser.add_argument(
        '--risk_averse_pct', type=float, default=0.33,
        help='Percentage of farmers that are risk averse (default: 0.33)'
    )
    parser.add_argument(
        '--risk_loving_pct', type=float, default=0.34,
        help='Percentage of farmers that are risk loving (default: 0.34)'
    )
    parser.add_argument(
        '--html_report', action='store_true',
        help='Generate HTML report with all plots'
    )
    parser.add_argument(
        '--math_equations', action='store_true',
        help='Generate HTML report with mathematical equations and visualizations'
    )
    return parser.parse_args()

def setup_model_parameters(args):
    """Set up model parameters based on command line arguments"""
    penalty_multiplier = args.penalty_multiplier
    testing_multiplier = args.testing_multiplier
    id_probability = args.identification_prob
    
    # Normalize risk percentages if they don't sum to 1
    total = args.risk_neutral_pct + args.risk_averse_pct + args.risk_loving_pct
    if abs(total - 1.0) > 1e-10:
        args.risk_neutral_pct /= total
        args.risk_averse_pct /= total
        args.risk_loving_pct /= total
        print(f"Note: Risk percentages normalized to sum to 1: "
              f"{args.risk_neutral_pct:.2f}, {args.risk_averse_pct:.2f}, {args.risk_loving_pct:.2f}")
    
    return {
        'num_farmers': args.num_farmers,
        'time_steps': args.time_steps,
        'seed': args.seed,
        'risk_neutral_pct': args.risk_neutral_pct,
        'risk_averse_pct': args.risk_averse_pct,
        'risk_loving_pct': args.risk_loving_pct,
        'penalty_multiplier': penalty_multiplier,
        'testing_multiplier': testing_multiplier,
        'id_probability': id_probability
    }

def ensure_directory(directory):
    """Ensure the output directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    """Main function to run the simulation."""
    args = parse_args()
    
    # Ensure output directory exists
    ensure_directory(args.output_dir)
    
    # Check if only generating math equations report without running simulation
    if args.math_equations and not args.analyze and not args.html_report:
        print("Generating mathematical equations report...")
        equations_dir = os.path.join(args.output_dir, 'equations')
        generate_equation_html_report(equations_dir)
        print(f"Mathematical equations report generated at {os.path.join(equations_dir, 'mathematical_foundations.html')}")
        print(f"View it in your browser by opening: file://{os.path.abspath(os.path.join(equations_dir, 'mathematical_foundations.html'))}")
        return
    
    # Set up model parameters
    params = setup_model_parameters(args)
    
    print(f"Starting Farmer's Risk Control Behaviors Simulation with {args.num_farmers} farmers")
    print(f"Running for {args.time_steps} time steps")
    print(f"Penalty multiplier: {args.penalty_multiplier}")
    print(f"Testing probability multiplier: {args.testing_multiplier}")
    print(f"Identification probability: {args.identification_prob}")
    print(f"Risk neutral percentage: {args.risk_neutral_pct}")
    print(f"Risk averse percentage: {args.risk_averse_pct}")
    print(f"Risk loving percentage: {args.risk_loving_pct}")
    
    # Create model
    model = FarmerRiskControlModel(
        num_farmers=params['num_farmers'],
        time_steps=params['time_steps'],
        seed=params['seed'],
        risk_neutral_pct=params['risk_neutral_pct'],
        risk_averse_pct=params['risk_averse_pct'],
        risk_loving_pct=params['risk_loving_pct'],
        penalty_multiplier=params['penalty_multiplier'],
        testing_multiplier=params['testing_multiplier'],
        id_probability=params['id_probability']
    )
    
    # Print out number of farmers by risk type
    print(f"Created {model.num_risk_neutral} risk neutral farmers")
    print(f"Created {model.num_risk_averse} risk averse farmers")
    print(f"Created {model.num_risk_loving} risk loving farmers")
    
    # Run model
    print("Running simulation...")
    model.run_simulation()
    
    # Save results
    model.save_results_to_file(args.output_dir)
    
    # Run detailed analysis and visualization if requested
    if args.analyze or args.html_report:
        print("Running detailed analysis and visualization...")
        from abm.visualize_results import analyze_model_results, generate_html_report
        
        # Run analysis
        analyze_model_results(model, args.output_dir)
        
        # Generate HTML report if requested
        if args.html_report:
            report_path = os.path.join(args.output_dir, "report.html")
            generate_html_report(args.output_dir)
            print(f"HTML report generated at {report_path}")
    else:
        print("Plotting basic results...")
        model.plot_results(args.output_dir)
    
    # Generate mathematical equations report if requested
    if args.math_equations:
        print("Generating mathematical equations report...")
        equations_dir = os.path.join(args.output_dir, 'equations')
        generate_equation_html_report(equations_dir)
        print(f"Mathematical equations report generated at {os.path.join(equations_dir, 'mathematical_foundations.html')}")
    
    print("Simulation complete!")
    print(f"Results saved in the '{args.output_dir}' directory")
    
    if args.html_report or args.analyze:
        html_path = os.path.join(args.output_dir, 'report.html')
        print(f"HTML report available at: {html_path}")
        print(f"View it in your browser by opening: file://{os.path.abspath(html_path)}")
        
    if args.math_equations:
        math_html_path = os.path.join(args.output_dir, 'equations', 'mathematical_foundations.html')
        print(f"Mathematical equations report available at: {math_html_path}")
        print(f"View it in your browser by opening: file://{os.path.abspath(math_html_path)}")

if __name__ == "__main__":
    main() 
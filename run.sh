#!/bin/bash

# Script to run the Farmer's Risk Control Behaviors ABM

# Set up the environment
echo "Setting up environment..."
pip install -r requirements.txt

# Create output directory
mkdir -p results

# Run the model with different parameter settings
echo "Running basic simulation..."
python run_farmer_risk_control_model.py --num_farmers 100 --time_steps 50 --seed 42 --output_dir results/basic

echo "Running high penalty simulation..."
python run_farmer_risk_control_model.py --num_farmers 100 --time_steps 50 --seed 42 --output_dir results/high_penalty --penalty_multiplier 2.0

echo "Running high testing simulation..."
python run_farmer_risk_control_model.py --num_farmers 100 --time_steps 50 --seed 42 --output_dir results/high_testing --testing_multiplier 2.0

echo "Running low identification probability simulation..."
python run_farmer_risk_control_model.py --num_farmers 100 --time_steps 50 --seed 42 --output_dir results/low_id_prob --identification_prob 0.2

echo "Running detailed analysis on a smaller simulation..."
python run_farmer_risk_control_model.py --num_farmers 50 --time_steps 20 --seed 42 --output_dir results/detailed_analysis --analysis

echo "All simulations completed successfully!"
echo "Results are available in the 'results' directory." 
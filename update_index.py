#!/usr/bin/env python3
"""
Script to update the results/index.html file with links to all simulation results.
This will scan the results directory for subdirectories and create a card for each one.
"""

import os
import sys
from abm.visualize_results import generate_html_report

def ensure_directory(directory):
    """Ensure the output directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def update_main_index(results_dir='results'):
    """
    Update the main index.html file with links to all simulation results.
    
    Parameters:
    -----------
    results_dir : str, optional
        Main results directory (default: 'results')
    """
    # Ensure results directory exists
    ensure_directory(results_dir)
    
    # Find all subdirectories in the results directory
    subdirs = [d for d in os.listdir(results_dir) 
             if os.path.isdir(os.path.join(results_dir, d)) and not d.startswith('.')]
    
    # Generate HTML report for each subdirectory if needed
    for subdir in subdirs:
        subdir_path = os.path.join(results_dir, subdir)
        report_path = os.path.join(subdir_path, 'report.html')
        
        # Generate report if it doesn't exist
        if not os.path.exists(report_path):
            try:
                print(f"Generating report for {subdir}...")
                generate_html_report(subdir_path)
            except Exception as e:
                print(f"Error generating report for {subdir}: {e}")
    
    # Create the main index.html content
    html_content = """<!DOCTYPE html>
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
        h1 {
            color: #0056b3;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }
        .scenario-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .scenario-card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 20px;
            transition: transform 0.3s ease;
        }
        .scenario-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .scenario-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #0056b3;
        }
        .scenario-description {
            margin-bottom: 15px;
            font-size: 14px;
            color: #666;
        }
        .scenario-link {
            display: inline-block;
            background-color: #0056b3;
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            text-decoration: none;
            font-size: 14px;
            transition: background-color 0.3s ease;
        }
        .scenario-link:hover {
            background-color: #003d82;
        }
    </style>
</head>
<body>
    <h1>Farmer Risk Control Behaviors - Simulation Results</h1>
    
    <div class="scenario-grid">
"""
    
    # Add a card for each subdirectory
    for subdir in sorted(subdirs):
        # Create a nicely formatted title from the directory name
        scenario_name = subdir.replace('_', ' ').title()
        
        # Try to get a description from the parameters file
        description = "Simulation scenario"
        params_file = os.path.join(results_dir, subdir, 'parameters.txt')
        if os.path.exists(params_file):
            try:
                with open(params_file, 'r') as f:
                    params = f.readlines()
                # Extract key information for the description
                farmer_count = next((p for p in params if "Number of farmers" in p), "").strip()
                risk_types = [p.strip() for p in params if "Risk" in p and "percentage" not in p][:3]
                testing = next((p for p in params if "Testing probabilities" in p), "").strip()
                
                # Create a meaningful description
                description_parts = []
                if farmer_count:
                    description_parts.append(farmer_count)
                if risk_types:
                    risk_types_str = ", ".join(risk_types)
                    description_parts.append(risk_types_str)
                if testing:
                    description_parts.append(testing)
                
                if description_parts:
                    description = " ".join(description_parts)
            except Exception as e:
                print(f"Error reading parameters for {subdir}: {e}")
        
        # Add the card to the HTML
        html_content += f"""        <div class="scenario-card">
            <div class="scenario-title">{scenario_name}</div>
            <div class="scenario-description">
                {description}
            </div>
            <a href="{subdir}/report.html" class="scenario-link">View Results</a>
        </div>
"""
    
    # Complete the HTML content
    html_content += """    </div>
</body>
</html>
"""
    
    # Write the HTML file
    index_file = os.path.join(results_dir, 'index.html')
    with open(index_file, 'w') as f:
        f.write(html_content)
    
    print(f"Index file updated: {index_file}")
    print(f"Found {len(subdirs)} simulation directories")

if __name__ == "__main__":
    # Use command line argument for results directory if provided
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'results'
    update_main_index(results_dir)
#!/bin/bash

# Script to run the Farmer's Risk Control Behaviors ABM with different scenarios

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -q -r requirements.txt

# Create output directories
echo "Creating output directories..."
mkdir -p results/baseline
mkdir -p results/mostly_risk_averse
mkdir -p results/mostly_risk_loving
mkdir -p results/mostly_risk_neutral
mkdir -p results/high_penalty
mkdir -p results/high_testing
mkdir -p results/low_id_prob
mkdir -p results/detailed

# Run the simulations with different scenarios
echo "Running baseline scenario (equal distribution of risk types)..."
python3 run_farmer_risk_control_model.py --num_farmers 100 --time_steps 50 --seed 42 \
    --output_dir results/baseline --risk_neutral_pct 0.33 --risk_averse_pct 0.33 --risk_loving_pct 0.34 \
    --penalty_multiplier 1.0 --testing_multiplier 1.0 --id_probability 0.5 --html_report

echo "Running mostly risk averse scenario..."
python3 run_farmer_risk_control_model.py --num_farmers 100 --time_steps 50 --seed 42 \
    --output_dir results/mostly_risk_averse --risk_neutral_pct 0.2 --risk_averse_pct 0.6 --risk_loving_pct 0.2 \
    --penalty_multiplier 1.0 --testing_multiplier 1.0 --id_probability 0.5 --html_report

echo "Running mostly risk loving scenario..."
python3 run_farmer_risk_control_model.py --num_farmers 100 --time_steps 50 --seed 42 \
    --output_dir results/mostly_risk_loving --risk_neutral_pct 0.2 --risk_averse_pct 0.2 --risk_loving_pct 0.6 \
    --penalty_multiplier 1.0 --testing_multiplier 1.0 --id_probability 0.5 --html_report

echo "Running mostly risk neutral scenario..."
python3 run_farmer_risk_control_model.py --num_farmers 100 --time_steps 50 --seed 42 \
    --output_dir results/mostly_risk_neutral --risk_neutral_pct 0.6 --risk_averse_pct 0.2 --risk_loving_pct 0.2 \
    --penalty_multiplier 1.0 --testing_multiplier 1.0 --id_probability 0.5 --html_report

echo "Running high penalty scenario..."
python3 run_farmer_risk_control_model.py --num_farmers 100 --time_steps 50 --seed 42 \
    --output_dir results/high_penalty --risk_neutral_pct 0.33 --risk_averse_pct 0.33 --risk_loving_pct 0.34 \
    --penalty_multiplier 2.0 --testing_multiplier 1.0 --id_probability 0.5 --html_report

echo "Running high testing scenario..."
python3 run_farmer_risk_control_model.py --num_farmers 100 --time_steps 50 --seed 42 \
    --output_dir results/high_testing --risk_neutral_pct 0.33 --risk_averse_pct 0.33 --risk_loving_pct 0.34 \
    --penalty_multiplier 1.0 --testing_multiplier 2.0 --id_probability 0.5 --html_report

echo "Running low identification probability scenario..."
python3 run_farmer_risk_control_model.py --num_farmers 100 --time_steps 50 --seed 42 \
    --output_dir results/low_id_prob --risk_neutral_pct 0.33 --risk_averse_pct 0.33 --risk_loving_pct 0.34 \
    --penalty_multiplier 1.0 --testing_multiplier 1.0 --id_probability 0.2 --html_report

echo "Running detailed analysis with longer simulation..."
python3 run_farmer_risk_control_model.py --num_farmers 150 --time_steps 100 --seed 42 \
    --output_dir results/detailed --risk_neutral_pct 0.33 --risk_averse_pct 0.33 --risk_loving_pct 0.34 \
    --penalty_multiplier 1.0 --testing_multiplier 1.0 --id_probability 0.5 --html_report --analyze

# Create an index.html file to navigate between the results
echo "Creating index.html to navigate between scenarios..."
cat > results/index.html << EOL
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
        <div class="scenario-card">
            <div class="scenario-title">Baseline Scenario</div>
            <div class="scenario-description">
                Equal distribution of risk preferences (33% each type).
                Standard penalty and testing probabilities.
            </div>
            <a href="baseline/report.html" class="scenario-link">View Results</a>
        </div>
        
        <div class="scenario-card">
            <div class="scenario-title">Mostly Risk Averse</div>
            <div class="scenario-description">
                60% risk averse farmers, 20% risk neutral, 20% risk loving.
                Standard penalty and testing probabilities.
            </div>
            <a href="mostly_risk_averse/report.html" class="scenario-link">View Results</a>
        </div>
        
        <div class="scenario-card">
            <div class="scenario-title">Mostly Risk Loving</div>
            <div class="scenario-description">
                60% risk loving farmers, 20% risk neutral, 20% risk averse.
                Standard penalty and testing probabilities.
            </div>
            <a href="mostly_risk_loving/report.html" class="scenario-link">View Results</a>
        </div>
        
        <div class="scenario-card">
            <div class="scenario-title">Mostly Risk Neutral</div>
            <div class="scenario-description">
                60% risk neutral farmers, 20% risk averse, 20% risk loving.
                Standard penalty and testing probabilities.
            </div>
            <a href="mostly_risk_neutral/report.html" class="scenario-link">View Results</a>
        </div>
        
        <div class="scenario-card">
            <div class="scenario-title">High Penalty</div>
            <div class="scenario-description">
                Equal distribution of risk preferences.
                Double penalty values (2x) for violations.
            </div>
            <a href="high_penalty/report.html" class="scenario-link">View Results</a>
        </div>
        
        <div class="scenario-card">
            <div class="scenario-title">High Testing</div>
            <div class="scenario-description">
                Equal distribution of risk preferences.
                Double testing probability (2x) at all test points.
            </div>
            <a href="high_testing/report.html" class="scenario-link">View Results</a>
        </div>
        
        <div class="scenario-card">
            <div class="scenario-title">Low Identification Probability</div>
            <div class="scenario-description">
                Equal distribution of risk preferences.
                Low probability (0.2) of identifying eligible products.
            </div>
            <a href="low_id_prob/report.html" class="scenario-link">View Results</a>
        </div>
        
        <div class="scenario-card">
            <div class="scenario-title">Detailed Analysis</div>
            <div class="scenario-description">
                Longer simulation with 150 farmers over 100 time steps.
                Includes comprehensive analysis and visualizations.
            </div>
            <a href="detailed/report.html" class="scenario-link">View Results</a>
        </div>
    </div>
</body>
</html>
EOL

echo "All simulations completed! Results available in the 'results' directory."
echo "Open results/index.html in your browser to view and navigate between different scenarios."
echo "You can use the following URL to access the index: file://$(pwd)/results/index.html"

# Deactivate virtual environment
deactivate 
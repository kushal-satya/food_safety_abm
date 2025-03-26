# US Fresh Produce Food Safety Testing Simulation 

## Introduction
This repository contains the code for a study on the risk management strategy of food safety in the US fresh produce supply chain using an agent-based model. The study identifies testing strategies using modelling to optimize for the overall system costs by considering all stakeholders.

## Model 3: Agent-Based Modeling of Farmer's Risk Control Behaviors
This model implements an agent-based simulation of farmers' risk control behaviors in a food safety context. It explores how farmers adjust their risk control efforts in response to testing regimes, penalties, and technology levels.

### Key Features
- **Exponential Function for Risk Control**: Uses an exponential function to model how farmer's risk control effort affects contamination rates.
- **Cost Optimization**: Farmers optimize their risk control efforts to minimize costs.
- **Multiple Test Points**: Includes penalties for contamination detection at four test points.
- **Farmer Heterogeneity**: Models different technology levels and initial risk control efforts across farmers.
- **Network Effects**: Includes potential influence from neighboring farmers.

### Mathematical Model
The model implements the equations described in Model 3:
1. **Contamination Rate Equation (3)**: σᵢᵗ = e^(-c_j^t * k_j)
2. **Cost Function Equation (4)**: Complex cost function considering testing probabilities, penalties, and costs
3. **Optimal Contamination Rate Equation (5)**: Farmers optimize risk control efforts based on cost minimization

## File Structure

- **/data**: Contains data files
  - raw_data.csv
  - processed_data.csv
- **/src**: Contains source code
- **/old_archives**: Contains Working Code/Files 
- **/abm**: Contains agent-based modeling files
  - **farmer_risk_control_model.py**: Implementation of Model 3
  - model.py
  - agent.py
  - supply.py
  - monthly.py
- main.py
- **run_farmer_risk_control_model.py**: Script to run the farmer risk control model
- model.ipynb
- requirements.txt
- abm.ipynb

## How to use

1. Install the required packages:
```
pip install -r requirements.txt
```

2. Run the farmer risk control model:
```
python run_farmer_risk_control_model.py --num_farmers 100 --time_steps 50 --seed 42
```

### Command-line options
- `--num_farmers`: Number of farmer agents (default: 100)
- `--time_steps`: Number of time steps to simulate (default: 50)
- `--seed`: Random seed for reproducibility (default: None)
- `--output_dir`: Directory to save results (default: results)

## Results
The simulation produces the following output files:
- `simulation_results.png`: Plots of mean risk control effort, contamination rate, and cost over time
- `alpha_distribution.png`: Distribution of final risk control efforts across farmers
- `individual_trajectories.png`: Risk control effort trajectories for a sample of individual farmers


# Agent-Based Modeling of Farmer's Risk Control Behaviors

## Introduction
This repository contains the code for an agent-based model (ABM) that simulates the risk management strategies of farmers in a food safety context. The model focuses on how farmers with different risk preferences adjust their risk control efforts over time in response to testing regimes, penalties, and technology levels.

Food safety cannot be completely guaranteed through an analysis of the final food product but needs to build on appropriate control measures throughout the food chain (Fritz and Schiefer 2009). This model simulates the pace of the evolution of farmers' risk control behavior under given environmental conditions.

## Model Description

### Model 3: Agent-Based Model of Farmer's Risk Control Behaviors

The model implements three types of farmers with different risk preferences:
1. **Risk Neutral Farmers**: Make decisions based on expected values without bias
2. **Risk Averse Farmers**: Are more concerned about potential penalties and invest more in risk control
3. **Risk Loving Farmers**: Are less concerned about penalties and invest less in risk control

### Key Features
- **Exponential Function for Risk Control**: Uses an exponential function to model how farmer's risk control effort affects contamination rates
- **Cost Optimization**: Farmers optimize their risk control efforts to minimize costs
- **Multiple Test Points**: Includes penalties for contamination detection at four test points
- **Farmer Heterogeneity**: Models different technology levels and initial risk control efforts across farmers
- **Network Effects**: Includes potential influence from neighboring farmers
- **Technology Adaptation**: Farmers can adjust their technology level over time

### Mathematical Model

The model is based on the following key equations:

#### Contamination Rate (Equation 3)
The contamination rate of a farmer is determined by their risk control effort and technology level:

$$\sigma_j^t = e^{-c_j^t \cdot k_j}$$

Where:
- $\sigma_j^t$ is the contamination rate of farmer $j$ at time $t$
- $c_j^t$ is the risk control effort of farmer $j$ at time $t$ (value between 0 and 1)
- $k_j$ is the technology level of farmer $j$ (value between 0 and 1)

#### Cost Function (Equation 4)
A farmer's cost function includes the contamination penalty due to resulting contamination at test points, risk control effort costs, and technology adaptation costs:

$$f = \frac{f_1\beta_1 + f_2(1-\beta_1)\beta_2 + f_3(1-\beta_1)(1-\beta_2)\beta_3 + f_4(1-\beta_1)(1-\beta_2)(1-\beta_3)\beta_4 + f_5(1-\beta_1)(1-\beta_2)(1-\beta_3)(1-\beta_4)P}{(c_e+c_k)(1-\beta_1)(1-\beta_2)(1-\beta_3)(1-\beta_4)}$$

Where:
- $f_1, f_2, f_3, f_4$ are penalties for contamination detected at test points 1-4
- $f_5$ is the penalty if contamination causes illness in consumers
- $\beta_1, \beta_2, \beta_3, \beta_4$ represent the testing probabilities at test points 1-4
- $P$ is the probability that the farmer's eligible products can be identified through tracing
- $c_e$ and $c_k$ are the effort costs and technology adaptation costs respectively

#### Optimal Risk Control Effort (Equation 5)
A farmer's optimal contamination rate can be found by minimizing the cost function (Eq. 4).

### Risk Preference Implementation

The model modifies how farmers perceive and respond to penalties based on their risk preference:

- **Risk Neutral Farmers**: Use the standard cost function calculation
- **Risk Averse Farmers**: Perceive penalties ($f_1$ to $f_5$) as higher by a factor of $(1 + risk\_coefficient \times 0.5)$
- **Risk Loving Farmers**: Perceive penalties ($f_1$ to $f_5$) as lower by a factor of $(1 - risk\_coefficient \times 0.3)$

## File Structure

- **/abm**: Contains the agent-based modeling files
  - **farmer_risk_control_model.py**: Implementation of Model 3 with the Farmer and FarmerRiskControlModel classes
  - **visualize_results.py**: Advanced visualization and analysis tools
- **run_farmer_risk_control_model.py**: Script to run the model with command-line options
- **requirements.txt**: Required Python packages
- **run.sh**: Bash script to run multiple simulations with different parameters

## How to use

### Installation

1. Create a virtual environment and install the required packages:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the model

2. Run the model with default parameters:
```bash
python run_farmer_risk_control_model.py
```

3. Run with custom parameters:
```bash
python run_farmer_risk_control_model.py --num_farmers 100 --time_steps 50 --seed 42 --output_dir results/custom --analysis
```

4. Run multiple scenarios using the run.sh script:
```bash
chmod +x run.sh
./run.sh
```

### Command-line options

- `--num_farmers`: Number of farmer agents (default: 100)
- `--time_steps`: Number of time steps to simulate (default: 50)
- `--seed`: Random seed for reproducibility (default: None)
- `--output_dir`: Directory to save results (default: results)
- `--analysis`: Run detailed analysis and visualization after simulation
- `--penalty_multiplier`: Multiplier for penalty values (default: 1.0)
- `--testing_multiplier`: Multiplier for testing probabilities (default: 1.0)
- `--identification_prob`: Probability of identifying eligible products (default: 0.5)
- `--risk_neutral_pct`: Percentage of farmers that are risk neutral (default: 0.33)
- `--risk_averse_pct`: Percentage of farmers that are risk averse (default: 0.33)
- `--risk_loving_pct`: Percentage of farmers that are risk loving (default: 0.34)
- `--html_report`: Generate an interactive HTML report with all plots

## Results and Analysis

The simulation produces various outputs:

### Main Results Files
- `simulation_results.png`: Plots of mean risk control effort, contamination rate, and cost over time
- `alpha_distribution_by_risk.png`: Distribution of final risk control efforts across farmers by risk type
- `technology_vs_contamination.png`: Relationship between technology level and contamination rate
- `technology_over_time.png`: Technology level evolution over time by risk preference
- `individual_trajectories_0.png`: Risk control effort trajectories for risk neutral farmers
- `individual_trajectories_1.png`: Risk control effort trajectories for risk averse farmers
- `individual_trajectories_2.png`: Risk control effort trajectories for risk loving farmers

### Advanced Analysis
- `effort_heatmap.png`: Heatmap of risk control effort evolution for all farmers
- `farmer_network.png`: Network visualization of farmers and their risk control decisions
- `effort_distribution_over_time.png`: Evolution of the distribution of risk control efforts

### Interactive Report
When the `--html_report` option is used, the model generates an interactive HTML report (`report.html`) that includes all plots with detailed explanations.

## Model Implementation Details

### Farmer Class
The `Farmer` class represents individual farmers with the following attributes:
- `id`: Unique identifier
- `alpha`: Risk control effort (0-1)
- `technology_level`: Technology level (0-1)
- `risk_preference`: One of RISK_NEUTRAL, RISK_AVERSE, or RISK_LOVING
- `risk_coefficient`: Strength of risk preference

Key methods:
- `calculate_contamination_rate()`: Implements equation 3
- `calculate_cost()`: Implements equation 4 with risk preference adjustments
- `find_optimal_contamination_rate()`: Uses optimization to find the optimal risk control effort
- `update_technology()`: Updates technology level based on experience and neighbors
- `update()`: Main update method for each time step

### FarmerRiskControlModel Class
The `FarmerRiskControlModel` class manages the simulation with the following features:
- Creates farmers with different risk preferences
- Runs the simulation for specified time steps
- Tracks metrics by risk preference type
- Generates visualizations of results

## References

1. Fritz, M., & Schiefer, G. (2009). Tracking, tracing, and business process interests in food commodities: A multi-level decision complexity. International Journal of Production Economics, 117(2), 317-329.
2. Ge et al. (2015a, 2015b; Ge et al. 2016). Studies on optimal food safety control strategies.



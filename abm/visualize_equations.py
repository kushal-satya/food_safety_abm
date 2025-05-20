import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
from abm.farmer_risk_control_model import Farmer, FarmerRiskControlModel

def ensure_directory(directory):
    """Ensure a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_equation_html_report(output_dir='results/equations'):
    """
    Generate an HTML report with mathematical equations and explanatory visualizations.
    
    Parameters:
    -----------
    output_dir : str, optional
        Directory to save the HTML report and associated images
    """
    # Create output directory if it doesn't exist
    ensure_directory(output_dir)
    
    # Create model to generate equation plots
    model = FarmerRiskControlModel(num_farmers=10, time_steps=10, seed=42)
    model.plot_mathematical_equations(output_dir)
    
    # Create HTML report
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Food Safety Risk Control Model - Mathematical Foundations</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
                color: #333;
                line-height: 1.6;
            }
            h1, h2, h3 {
                color: #0056b3;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                border-radius: 5px;
            }
            .equation {
                background-color: #f8f9fa;
                padding: 15px;
                border-left: 4px solid #0056b3;
                margin: 20px 0;
                font-family: "Times New Roman", serif;
                font-style: italic;
            }
            .plot {
                width: 100%;
                max-width: 800px;
                margin: 20px auto;
                display: block;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .description {
                margin: 20px 0;
            }
            .equation-number {
                float: right;
                font-weight: bold;
            }
            .parameter-table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            .parameter-table th, .parameter-table td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            .parameter-table th {
                background-color: #f2f2f2;
            }
            .risk-comparison {
                display: flex;
                justify-content: space-between;
                margin: 20px 0;
            }
            .risk-type {
                flex: 1;
                padding: 15px;
                margin: 0 10px;
                border-radius: 5px;
            }
            .risk-neutral {
                background-color: #e6f2ff;
                border: 1px solid #99ccff;
            }
            .risk-averse {
                background-color: #e6ffe6;
                border: 1px solid #99ff99;
            }
            .risk-loving {
                background-color: #ffe6e6;
                border: 1px solid #ff9999;
            }
            .toc {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 30px;
            }
            .toc ul {
                list-style-type: none;
                padding-left: 20px;
            }
            .toc li {
                margin: 8px 0;
            }
            .toc a {
                text-decoration: none;
                color: #0056b3;
            }
            .toc a:hover {
                text-decoration: underline;
            }
        </style>
        <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
        <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    </head>
    <body>
        <div class="container">
            <h1>Food Safety Risk Control Model - Mathematical Foundations</h1>
            
            <div class="toc">
                <h2>Table of Contents</h2>
                <ul>
                    <li><a href="#introduction">1. Introduction</a></li>
                    <li><a href="#risk-types">2. Farmer Risk Preferences</a></li>
                    <li><a href="#contamination">3. Contamination Rate Function</a></li>
                    <li><a href="#cost">4. Cost Function</a></li>
                    <li><a href="#optimal">5. Optimal Risk Control Effort</a></li>
                    <li><a href="#parameters">6. Model Parameters</a></li>
                    <li><a href="#implementation">7. Implementation Details</a></li>
                </ul>
            </div>
            
            <section id="introduction">
                <h2>1. Introduction</h2>
                <p class="description">
                    This document presents the mathematical foundations of the Agent-Based Model (ABM) for 
                    food safety risk control behaviors. The model simulates how farmers with different risk 
                    preferences make decisions about their risk control efforts and technology adoption in 
                    response to testing regimes, penalties, and market conditions.
                </p>
                <p class="description">
                    Each farmer agent in the model aims to minimize their cost function, which includes 
                    both direct costs (effort and technology) and expected penalty costs from contamination. 
                    The model captures how different risk preferences lead to different decision-making 
                    strategies and outcomes.
                </p>
            </section>
            
            <section id="risk-types">
                <h2>2. Farmer Risk Preferences</h2>
                <p class="description">
                    The model includes three types of farmers with different risk preferences:
                </p>
                
                <div class="risk-comparison">
                    <div class="risk-type risk-neutral">
                        <h3>Risk Neutral</h3>
                        <p>Risk neutral farmers weigh costs and benefits exactly as they are. They don't 
                        overestimate or underestimate risk probabilities or penalties.</p>
                        <p><strong>Cost perception:</strong> Unmodified</p>
                        <p><strong>Risk perception:</strong> Accurate</p>
                    </div>
                    
                    <div class="risk-type risk-averse">
                        <h3>Risk Averse</h3>
                        <p>Risk averse farmers are more sensitive to potential penalties and contamination risks. 
                        They tend to invest more in risk control efforts and technology.</p>
                        <p><strong>Cost perception:</strong> Penalties perceived as higher</p>
                        <p><strong>Risk perception:</strong> Overestimates risks</p>
                    </div>
                    
                    <div class="risk-type risk-loving">
                        <h3>Risk Loving</h3>
                        <p>Risk loving farmers are less sensitive to potential penalties and more willing to 
                        take chances. They tend to invest less in risk control efforts.</p>
                        <p><strong>Cost perception:</strong> Penalties perceived as lower</p>
                        <p><strong>Risk perception:</strong> Underestimates risks</p>
                    </div>
                </div>
                
                <p class="description">
                    Risk preferences are implemented in the model by adjusting how farmers perceive costs 
                    and probabilities in their decision-making processes:
                </p>
                
                <img src="risk_preference_cost_effect.png" alt="Effect of Risk Preference on Cost Perception" class="plot">
                
                <p class="description">
                    The impact of risk preferences is further illustrated by examining how different farmer types 
                    choose their optimal risk control effort:
                </p>
                
                <img src="risk_preference_optimal_effort.png" alt="Effect of Risk Preference on Optimal Effort" class="plot">
                
                <p class="description">
                    As shown in the plot, risk averse farmers tend to choose higher levels of risk control effort 
                    (higher α) compared to risk neutral farmers, while risk loving farmers choose lower levels.
                </p>
            </section>
            
            <section id="contamination">
                <h2>3. Contamination Rate Function</h2>
                <p class="description">
                    The contamination rate for a farmer is modeled using an exponential function that 
                    depends on the farmer's risk control effort and technology level:
                </p>
                
                <div class="equation">
                    <span class="equation-number">(1)</span>
                    \\[ \sigma_j^t = e^{-c_j^t \cdot k_j} \\]
                </div>
                
                <p class="description">
                    where:
                </p>
                <ul>
                    <li>\( \sigma_j^t \) is the contamination rate for farmer \(j\) at time \(t\)</li>
                    <li>\( c_j^t \) is the risk control effort of farmer \(j\) at time \(t\)</li>
                    <li>\( k_j \) is the technology level of farmer \(j\)</li>
                </ul>
                
                <p class="description">
                    This exponential function has several important properties:
                </p>
                <ul>
                    <li>Higher risk control effort (\(c\)) leads to lower contamination rates</li>
                    <li>Higher technology level (\(k\)) makes effort more effective</li>
                    <li>There are diminishing marginal returns to increased effort</li>
                    <li>When \(c \cdot k \to 0\), \(\sigma \to 1\) (maximum contamination)</li>
                    <li>When \(c \cdot k \to \infty\), \(\sigma \to 0\) (minimum contamination)</li>
                </ul>
                
                <img src="equation3_contamination_rate.png" alt="Contamination Rate Function" class="plot">
                
                <p class="description">
                    The plot shows how contamination rate decreases as risk control effort increases, 
                    and how higher technology levels make the same effort more effective at reducing contamination.
                </p>
            </section>
            
            <section id="cost">
                <h2>4. Cost Function</h2>
                <p class="description">
                    A farmer's cost function includes both direct costs (effort and technology) and expected 
                    penalty costs from contamination detection:
                </p>
                
                <div class="equation">
                    <span class="equation-number">(2)</span>
                    \\[ Cost = (c_e \cdot c_j^t + c_k \cdot k_j) + \frac{ExpectedPenalty}{PassProbability} \\]
                </div>
                
                <p class="description">
                    where:
                </p>
                <ul>
                    <li>\( c_e \) is the cost per unit of effort</li>
                    <li>\( c_k \) is the cost per unit of technology</li>
                    <li>\( ExpectedPenalty \) is the expected penalty from contamination detection</li>
                    <li>\( PassProbability \) is the probability of passing all tests</li>
                </ul>
                
                <p class="description">
                    The expected penalty component is calculated as:
                </p>
                
                <div class="equation">
                    <span class="equation-number">(3)</span>
                    \begin{align*}
                    ExpectedPenalty = & f_1 \cdot \beta_1 \cdot \sigma + \\
                    & f_2 \cdot (1-\beta_1) \cdot \beta_2 \cdot \sigma + \\
                    & f_3 \cdot (1-\beta_1) \cdot (1-\beta_2) \cdot \beta_3 \cdot \sigma + \\
                    & f_4 \cdot (1-\beta_1) \cdot (1-\beta_2) \cdot (1-\beta_3) \cdot \beta_4 \cdot \sigma + \\
                    & f_5 \cdot (1-\beta_1) \cdot (1-\beta_2) \cdot (1-\beta_3) \cdot (1-\beta_4) \cdot P \cdot \sigma
                    \end{align*}
                </div>
                
                <p class="description">
                    where:
                </p>
                <ul>
                    <li>\( f_1 \) to \( f_5 \) are the penalties at test points 1-4 and from illness</li>
                    <li>\( \beta_1 \) to \( \beta_4 \) are the testing probabilities at test points 1-4</li>
                    <li>\( P \) is the probability that a farmer's eligible products can be identified through tracing</li>
                    <li>\( \sigma \) is the contamination rate</li>
                </ul>
                
                <p class="description">
                    The pass probability is calculated as:
                </p>
                
                <div class="equation">
                    <span class="equation-number">(4)</span>
                    \\[ PassProbability = (1-\beta_1) \cdot (1-\beta_2) \cdot (1-\beta_3) \cdot (1-\beta_4) \\]
                </div>
                
                <img src="equation4_cost_function.png" alt="Cost Function Components" class="plot">
                
                <p class="description">
                    The plot shows how direct costs increase linearly with effort, while expected penalties 
                    decrease exponentially as effort increases (due to lower contamination). The total cost 
                    function has a U-shape, indicating there is an optimal level of effort that minimizes total cost.
                </p>
            </section>
            
            <section id="optimal">
                <h2>5. Optimal Risk Control Effort</h2>
                <p class="description">
                    A farmer's optimal risk control effort (\(\alpha^*\)) is the level that minimizes the total cost:
                </p>
                
                <div class="equation">
                    <span class="equation-number">(5)</span>
                    \\[ \alpha^* = \arg\min_{\alpha} \: Cost(\alpha) \\]
                </div>
                
                <img src="equation5_optimal_effort.png" alt="Optimal Risk Control Effort" class="plot">
                
                <p class="description">
                    The plot shows how the total cost varies with different levels of risk control effort, 
                    and the point where the cost is minimized (indicated by the vertical red line).
                </p>
                
                <p class="description">
                    In the simulation, farmers try to find this optimal effort level, but their perception 
                    of costs and risks is influenced by their risk preferences. This leads to different 
                    optimal effort levels for farmers with different risk preferences.
                </p>
            </section>
            
            <section id="parameters">
                <h2>6. Model Parameters</h2>
                <p class="description">
                    The model includes several key parameters that influence farmer behavior and outcomes:
                </p>
                
                <table class="parameter-table">
                    <tr>
                        <th>Parameter</th>
                        <th>Description</th>
                        <th>Default Value</th>
                    </tr>
                    <tr>
                        <td>f<sub>1</sub> to f<sub>5</sub></td>
                        <td>Penalties at test points 1-4 and from illness</td>
                        <td>[100, 300, 600, 1000, 5000]</td>
                    </tr>
                    <tr>
                        <td>β<sub>1</sub> to β<sub>4</sub></td>
                        <td>Testing probabilities at test points 1-4</td>
                        <td>[0.1, 0.15, 0.2, 0.25]</td>
                    </tr>
                    <tr>
                        <td>P</td>
                        <td>Probability of identifying eligible products</td>
                        <td>0.5</td>
                    </tr>
                    <tr>
                        <td>c<sub>e</sub> range</td>
                        <td>Range for effort cost</td>
                        <td>(200, 500)</td>
                    </tr>
                    <tr>
                        <td>c<sub>k</sub> range</td>
                        <td>Range for technology cost</td>
                        <td>(500, 1000)</td>
                    </tr>
                    <tr>
                        <td>Risk coefficients</td>
                        <td>Strength of risk preference effect</td>
                        <td>Neutral: 1.0, Averse: 0.8-1.5, Loving: 0.5-1.2</td>
                    </tr>
                </table>
                
                <p class="description">
                    These parameters can be adjusted to simulate different policy scenarios and market conditions.
                </p>
            </section>
            
            <section id="implementation">
                <h2>7. Implementation Details</h2>
                <p class="description">
                    The model is implemented as an agent-based simulation with the following key components:
                </p>
                
                <h3>Farmer Class</h3>
                <p class="description">
                    The Farmer class represents individual farmers with attributes including:
                </p>
                <ul>
                    <li>Risk control effort (α)</li>
                    <li>Technology level (k)</li>
                    <li>Risk preference type (neutral, averse, or loving)</li>
                    <li>Risk coefficient (strength of risk preference)</li>
                    <li>History of contamination rates, costs, and decisions</li>
                </ul>
                
                <h3>FarmerRiskControlModel Class</h3>
                <p class="description">
                    The FarmerRiskControlModel class manages the simulation with:
                </p>
                <ul>
                    <li>Population of farmer agents</li>
                    <li>Testing and penalty parameters</li>
                    <li>Simulation time steps</li>
                    <li>Analysis and visualization methods</li>
                </ul>
                
                <p class="description">
                    In each time step, farmers:
                </p>
                <ol>
                    <li>Update their technology level based on own experience and neighbor influences</li>
                    <li>Update their risk control effort based on testing experience and contamination detection</li>
                    <li>Calculate their new contamination rate and costs</li>
                    <li>Make decisions to optimize their effort for the next time step</li>
                </ol>
                
                <p class="description">
                    The simulation tracks individual farmer trajectories as well as aggregate statistics 
                    by risk preference type, allowing for analysis of how different farmer types respond 
                    to various policy interventions and market conditions.
                </p>
            </section>
        </div>
    </body>
    </html>
    """
    
    # Write the HTML file
    with open(os.path.join(output_dir, 'mathematical_foundations.html'), 'w') as f:
        f.write(html_content)
    
    print(f"HTML mathematical foundations report generated at {os.path.join(output_dir, 'mathematical_foundations.html')}")

if __name__ == "__main__":
    generate_equation_html_report() 
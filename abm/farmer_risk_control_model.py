import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import pandas as pd

class Farmer:
    """
    Agent class representing a farmer in the food safety ABM model.
    This implements the risk control behavior model as specified in Model 3.
    """
    # Risk preference types
    RISK_NEUTRAL = 0
    RISK_AVERSE = 1
    RISK_LOVING = 2
    
    RISK_TYPE_NAMES = {
        RISK_NEUTRAL: "Risk Neutral",
        RISK_AVERSE: "Risk Averse",
        RISK_LOVING: "Risk Loving"
    }
    
    def __init__(self, farmer_id, time_steps, alpha=0.5, technology_level=0.5, risk_preference=RISK_NEUTRAL, risk_coefficient=1.0):
        """
        Initialize a farmer agent.
        
        Parameters:
        -----------
        farmer_id : int
            Unique identifier for the farmer
        time_steps : int
            Total number of time steps in the simulation
        alpha : float, optional
            Initial value for risk control effort (between 0 and 1)
        technology_level : float, optional
            Initial value for technology level (between 0 and 1)
        risk_preference : int, optional
            Risk preference type: RISK_NEUTRAL (0), RISK_AVERSE (1), or RISK_LOVING (2)
        risk_coefficient : float, optional
            Coefficient determining the strength of risk preference (higher = stronger effect)
        """
        self.id = farmer_id
        self.alpha = alpha  # Initial risk control effort
        self.technology_level = technology_level  # k parameter in the model
        self.contamination_rate = None  # Will be calculated using the exponential function
        self.risk_preference = risk_preference
        self.risk_coefficient = risk_coefficient
        
        # Initialize arrays to store history
        self.time_steps = time_steps
        self.alpha_history = np.zeros(time_steps)
        self.contamination_history = np.zeros(time_steps)
        self.cost_history = np.zeros(time_steps)
        self.technology_history = np.zeros(time_steps)
        
        # Cost components
        self.c_e = 0  # Effort cost
        self.c_k = 0  # Technology adaptation cost
        self.penalties = [0, 0, 0, 0, 0]  # f1, f2, f3, f4, f5
        self.testing_regimes = [0, 0, 0, 0]  # β1, β2, β3, β4
        
        # New parameters for tracking individual experience
        self.was_tested = False           # D₁ in Eq. 6 - whether farmer was tested in previous step
        self.contamination_detected = False  # D₂ in Eq. 6 - whether contamination was detected
        self.tech_innovation_adopted = False  # D₃ in Eq. 7 - technology innovation adoption
        
        # Parameters for behavioral equations
        self.lambda1 = np.random.uniform(0.01, 0.05)  # Impact of being tested on effort
        self.lambda2 = np.random.uniform(0.05, 0.15)  # Impact of detected contamination on effort
        self.lambda3 = np.random.uniform(0.01, 0.05)  # Impact of peer contamination on effort
        
        self.omega1 = np.random.uniform(0.02, 0.08)  # Impact of own contamination on technology
        self.omega2 = np.random.uniform(0.01, 0.05)  # Impact of peer contamination on technology
        self.omega3 = np.random.uniform(0.05, 0.15)  # Impact of innovation adoption on technology
        
    def calculate_contamination_rate(self, t):
        """
        Calculate contamination rate using the exponential function from Eq. (3)
        σ_j^t = e^(-c_j^t * k_j)
        """
        c_j_t = self.alpha
        k_j = self.technology_level
        return np.exp(-c_j_t * k_j)
    
    def apply_risk_preference(self, cost):
        """
        Apply risk preference to the cost calculation.
        
        For risk neutral farmers: no change
        For risk averse farmers: higher cost (more sensitive to risk)
        For risk loving farmers: lower cost (less sensitive to risk)
        """
        if self.risk_preference == self.RISK_NEUTRAL:
            return cost
        elif self.risk_preference == self.RISK_AVERSE:
            # Risk averse farmers perceive costs as higher
            # More emphasis on penalties relative to effort/technology costs
            return cost * (1 + self.risk_coefficient * 0.5)
        elif self.risk_preference == self.RISK_LOVING:
            # Risk loving farmers perceive costs as lower
            # Less emphasis on penalties relative to effort/technology costs
            return cost * (1 - self.risk_coefficient * 0.3)
        else:
            return cost
    
    def calculate_cost(self, f, beta, P, c_e, c_k):
        """
        Calculate the cost function based on Eq. (4)
        f = [f1, f2, f3, f4, f5] : penalties at test points 1-4 and from illness
        beta = [β1, β2, β3, β4] : testing probabilities at test points 1-4
        P : probability that a farmer's eligible products can be identified
        c_e : effort cost
        c_k : technology cost
        """
        f1, f2, f3, f4, f5 = f
        beta1, beta2, beta3, beta4 = beta
        
        # Apply risk preference adjustment to penalties
        if self.risk_preference == self.RISK_AVERSE:
            # Risk averse farmers weigh penalties more heavily
            penalty_factor = 1 + self.risk_coefficient * 0.5
            f1 *= penalty_factor
            f2 *= penalty_factor
            f3 *= penalty_factor
            f4 *= penalty_factor
            f5 *= penalty_factor
            # Risk averse farmers also have higher perceived effort costs
            c_e = c_e * 0.8  # Lower perceived effort cost (more willing to put in effort)
        elif self.risk_preference == self.RISK_LOVING:
            # Risk loving farmers weigh penalties less heavily
            penalty_factor = 1 - self.risk_coefficient * 0.3
            f1 *= penalty_factor
            f2 *= penalty_factor
            f3 *= penalty_factor
            f4 *= penalty_factor
            f5 *= penalty_factor
            # Risk loving farmers have higher perceived effort costs
            c_e = c_e * 1.5  # Higher perceived effort cost (less willing to put in effort)
        
        # Calculate expected penalty (numerator in equation 4)
        expected_penalty = f1 * beta1 + \
                          f2 * (1 - beta1) * beta2 + \
                          f3 * (1 - beta1) * (1 - beta2) * beta3 + \
                          f4 * (1 - beta1) * (1 - beta2) * (1 - beta3) * beta4 + \
                          f5 * (1 - beta1) * (1 - beta2) * (1 - beta3) * (1 - beta4) * P
        
        # Calculate probability of passing all tests (denominator in equation 4)
        pass_probability = (1 - beta1) * (1 - beta2) * (1 - beta3) * (1 - beta4)
        
        # Direct costs of effort and technology
        direct_costs = (c_e * self.alpha + c_k * self.technology_level)
        
        # Store values for later reference
        self.c_e = c_e
        self.c_k = c_k
        self.penalties = [f1, f2, f3, f4, f5]
        self.testing_regimes = [beta1, beta2, beta3, beta4]
        
        if pass_probability == 0:
            return float('inf')
        
        # Total cost is direct costs plus expected penalty normalized by pass probability
        cost = direct_costs + (expected_penalty / pass_probability)
        
        return cost
    
    def update_technology(self, neighbors=None):
        """
        Update the farmer's technology level based on own experience,
        neighbor effects, and innovation adoption (Equation 7)
        
        Parameters:
        -----------
        neighbors : list of Farmer, optional
            List of neighboring farmers that may influence this farmer's technology
        """
        # Starting technology level (k_j^{T-1})
        current_tech = self.technology_level
        
        # Own contamination effect (ω₁D₂ⱼᵀ⁻¹)
        own_contamination_effect = self.omega1 * (1 if self.contamination_detected else 0)
        
        # Peer contamination effect (ω₂∑D₂ᵢᵀ⁻¹)
        peer_effect = 0
        if neighbors and len(neighbors) > 0:
            # Count neighbors with detected contamination
            detected_count = sum(1 for n in neighbors if n.contamination_detected)
            peer_effect = self.omega2 * detected_count
        
        # Innovation adoption effect (ω₃D₃ⱼᵀ)
        innovation_effect = self.omega3 * (1 if self.tech_innovation_adopted else 0)
        
        # Random small fluctuation to add realism
        random_adjustment = np.random.normal(0, 0.01)
        
        # Update technology level using Equation 7
        new_tech = current_tech + own_contamination_effect + peer_effect + innovation_effect + random_adjustment
        
        # Keep technology level within bounds
        self.technology_level = max(0.1, min(0.9, new_tech))
        
        # Reset flags for next time step
        # We'll set these in the run_simulation method based on testing outcomes
        self.tech_innovation_adopted = False
        
        return self.technology_level
    
    def find_optimal_contamination_rate(self, f, beta, P, c_e, c_k):
        """
        Find the optimal contamination rate by balancing cost of effort against expected penalties.
        Different farmer types have different alpha ranges they consider.
        """
        # Define different alpha ranges depending on risk preference
        if self.risk_preference == self.RISK_NEUTRAL:
            # Risk neutral farmers consider the full range
            test_alphas = np.linspace(0.05, 0.95, 30)
        elif self.risk_preference == self.RISK_AVERSE:
            # Risk averse farmers focus on higher effort values
            test_alphas = np.linspace(0.5, 0.95, 30)
        elif self.risk_preference == self.RISK_LOVING:
            # Risk loving farmers focus on lower effort values
            test_alphas = np.linspace(0.05, 0.7, 30)
        
        costs = []
        
        # Store current alpha
        current_alpha = self.alpha
        
        # Try each alpha value
        for alpha in test_alphas:
            self.alpha = alpha
            contamination = self.calculate_contamination_rate(0)  # Calculate contamination for this alpha
            
            # As contamination increases, so does the chance of failing tests
            # Modify testing probabilities based on contamination, with different adjustments by risk type
            if self.risk_preference == self.RISK_AVERSE:
                # Risk averse farmers overestimate test failure probabilities
                modified_beta = [
                    min(0.95, beta[0] * (1 + contamination * 1.5)),
                    min(0.95, beta[1] * (1 + contamination * 1.5)),
                    min(0.95, beta[2] * (1 + contamination * 1.5)), 
                    min(0.95, beta[3] * (1 + contamination * 1.5))
                ]
            elif self.risk_preference == self.RISK_LOVING:
                # Risk loving farmers underestimate test failure probabilities
                modified_beta = [
                    min(0.95, beta[0] * (1 + contamination * 0.5)),
                    min(0.95, beta[1] * (1 + contamination * 0.5)),
                    min(0.95, beta[2] * (1 + contamination * 0.5)), 
                    min(0.95, beta[3] * (1 + contamination * 0.5))
                ]
            else:
                # Risk neutral farmers have accurate assessment
                modified_beta = [
                    min(0.95, beta[0] * (1 + contamination)),
                    min(0.95, beta[1] * (1 + contamination)),
                    min(0.95, beta[2] * (1 + contamination)), 
                    min(0.95, beta[3] * (1 + contamination))
                ]
            
            cost = self.calculate_cost(f, modified_beta, P, c_e, c_k)
            costs.append(cost)
        
        # Find the alpha with the lowest cost
        if not all(np.isinf(costs)) and not all(np.isnan(costs)):
            valid_costs = [c if not np.isinf(c) and not np.isnan(c) else float('inf') for c in costs]
            best_idx = np.argmin(valid_costs)
            best_cost = valid_costs[best_idx]
            
            # Check if the best cost is valid
            if not np.isinf(best_cost):
                new_alpha = test_alphas[best_idx]
                
                # Add some randomness/inertia to decisions to create more realistic behavior
                # Different farmer types have different levels of adherence to optimal strategy
                if self.risk_preference == self.RISK_NEUTRAL:
                    # Risk neutral farmers are more consistent
                    adjustment = np.random.normal(0, 0.05)  # Small random adjustment
                    self.alpha = max(0.05, min(0.95, 0.8 * new_alpha + 0.2 * current_alpha + adjustment))
                elif self.risk_preference == self.RISK_AVERSE:
                    # Risk averse farmers are more conservative in changing their strategy
                    adjustment = np.random.normal(0, 0.03)  # Smaller random adjustment
                    self.alpha = max(0.05, min(0.95, 0.7 * new_alpha + 0.3 * current_alpha + adjustment))
                elif self.risk_preference == self.RISK_LOVING:
                    # Risk loving farmers are more volatile in their decisions
                    adjustment = np.random.normal(0, 0.1)  # Larger random adjustment
                    self.alpha = max(0.05, min(0.95, 0.6 * new_alpha + 0.4 * current_alpha + adjustment))
            else:
                self.alpha = current_alpha
        else:
            # If all costs are invalid, reset to original alpha
            self.alpha = current_alpha
        
        return self.alpha
    
    def update_alpha(self, neighbors=None, regional_benchmark=1):
        """
        Update the farmer's risk control effort based on testing experience,
        detected contamination, and neighbor effects (Equation 6)
        
        Parameters:
        -----------
        neighbors : list of Farmer, optional
            List of neighboring farmers that may influence this farmer's decisions
        regional_benchmark : float, optional
            Benchmark number/rate of detected contamination (h in Eq. 6)
        """
        # Current risk control effort (c_j^{T-1})
        current_alpha = self.alpha
        
        # Testing effect (λ₁D₁ⱼᵀ⁻¹)
        testing_effect = self.lambda1 * (1 if self.was_tested else 0)
        
        # Contamination detection effect (λ₂D₂ⱼᵀ⁻¹)
        detection_effect = self.lambda2 * (1 if self.contamination_detected else 0)
        
        # Peer/regional effect (λ₃∑(D₂ᵢᵀ⁻¹ - h))
        peer_effect = 0
        if neighbors and len(neighbors) > 0:
            # Calculate region contamination relative to benchmark
            detected_count = sum(1 for n in neighbors if n.contamination_detected)
            regional_effect = detected_count - regional_benchmark
            peer_effect = self.lambda3 * regional_effect
        
        # External shock/policy effect (δᵀ)
        policy_effect = 0  # Can be set to a non-zero value to simulate external policy change
        
        # Risk preference adjustment - different farmer types respond differently
        if self.risk_preference == self.RISK_AVERSE:
            # Risk averse farmers are more responsive to contamination
            detection_effect *= 1.5
            peer_effect *= 1.2
        elif self.risk_preference == self.RISK_LOVING:
            # Risk loving farmers are less responsive to testing and more resistant to effort increases
            testing_effect *= 0.7
            detection_effect *= 0.8
        
        # Random small fluctuation to add realism
        random_adjustment = np.random.normal(0, 0.01)
        
        # Update risk control effort using Equation 6
        new_alpha = current_alpha + testing_effect + detection_effect + peer_effect + policy_effect + random_adjustment
        
        # Keep alpha within bounds
        self.alpha = max(0.05, min(0.95, new_alpha))
        
        # Reset flags for next time step
        self.was_tested = False
        self.contamination_detected = False
        
        return self.alpha
    
    def update(self, t, f, beta, P, c_e, c_k, neighbors=None, regional_benchmark=1):
        """
        Update the farmer's risk control effort and contamination rate for time step t
        
        Parameters:
        -----------
        t : int
            Current time step
        f, beta, P, c_e, c_k : model parameters as described in calculate_cost
        neighbors : list of Farmer, optional
            List of neighboring farmers that may influence this farmer's decision
        regional_benchmark : float, optional
            Benchmark number/rate of detected contamination in region
        """
        # Two-step update process:
        # 1. Use equation-based update methods (Eq. 6 & 7) for individual evolution
        # 2. Fine-tune with optimization-based approach

        # Step 1: Update based on behavioral equations
        # Update technology level (Eq. 7)
        self.update_technology(neighbors)
        
        # Update risk control effort (Eq. 6)
        self.update_alpha(neighbors, regional_benchmark)
        
        # Step 2: Fine-tune with optimization (optional based on risk type)
        # Risk averse farmers rely more on equation-based updates (cautious)
        # Risk loving farmers optimize more aggressively (opportunistic)
        
        if self.risk_preference == self.RISK_NEUTRAL:
            # Balanced approach - equal weight to both mechanisms
            current_alpha = self.alpha
            optimal_alpha = self.find_optimal_contamination_rate(f, beta, P, c_e, c_k)
            self.alpha = 0.5 * current_alpha + 0.5 * optimal_alpha
        elif self.risk_preference == self.RISK_LOVING:
            # Risk lovers focus more on short-term optimization
            current_alpha = self.alpha
            optimal_alpha = self.find_optimal_contamination_rate(f, beta, P, c_e, c_k)
            self.alpha = 0.3 * current_alpha + 0.7 * optimal_alpha
        
        # Calculate contamination rate
        self.contamination_rate = self.calculate_contamination_rate(t)
        
        # Calculate cost
        cost = self.calculate_cost(f, beta, P, c_e, c_k)
        
        # Store history
        self.alpha_history[t] = self.alpha
        self.contamination_history[t] = self.contamination_rate
        self.cost_history[t] = cost
        self.technology_history[t] = self.technology_level
        
        return self.alpha, self.contamination_rate, cost


class FarmerRiskControlModel:
    """
    Agent-Based Model for simulating farmers' risk control behaviors.
    """
    def __init__(self, num_farmers=100, time_steps=50, seed=None,
                 risk_neutral_pct=0.33, risk_averse_pct=0.33, risk_loving_pct=0.34,
                 penalty_multiplier=1.0, testing_multiplier=1.0, id_probability=0.5,
                 initial_contamination_multiplier=1.0, testing_cost_per_sample=15.0):
        """
        Initialize the ABM.
        
        Parameters:
        -----------
        num_farmers : int, optional
            Number of farmer agents
        time_steps : int, optional
            Number of time steps to simulate
        seed : int, optional
            Random seed for reproducibility
        risk_neutral_pct : float, optional
            Percentage of farmers that are risk neutral (0-1)
        risk_averse_pct : float, optional
            Percentage of farmers that are risk averse (0-1)
        risk_loving_pct : float, optional
            Percentage of farmers that are risk loving (0-1)
        penalty_multiplier : float, optional
            Multiplier for penalty values
        testing_multiplier : float, optional
            Multiplier for testing probabilities
        id_probability : float, optional
            Probability that a farmer's eligible products can be identified
        initial_contamination_multiplier : float, optional
            Multiplier for initial contamination rates
        testing_cost_per_sample : float, optional
            Cost per test sample
        """
        if seed is not None:
            np.random.seed(seed)
            
        self.num_farmers = num_farmers
        self.time_steps = time_steps
        self.testing_cost_per_sample = testing_cost_per_sample
        
        # Ensure percentages sum to 1
        total = risk_neutral_pct + risk_averse_pct + risk_loving_pct
        if abs(total - 1.0) > 1e-10:
            risk_neutral_pct /= total
            risk_averse_pct /= total
            risk_loving_pct /= total
        
        # Calculate number of farmers in each category
        self.num_risk_neutral = int(num_farmers * risk_neutral_pct)
        self.num_risk_averse = int(num_farmers * risk_averse_pct)
        self.num_risk_loving = num_farmers - self.num_risk_neutral - self.num_risk_averse
        
        # Model parameters with more realistic values
        self.f = [
            100 * penalty_multiplier,  # f1: Penalty at test point 1
            300 * penalty_multiplier,  # f2: Penalty at test point 2
            600 * penalty_multiplier,  # f3: Penalty at test point 3
            1000 * penalty_multiplier, # f4: Penalty at test point 4
            5000 * penalty_multiplier  # f5: Penalty from illness
        ]
        
        # Testing probabilities are affected by testing_multiplier but capped at 0.95
        self.beta = [
            min(0.95, 0.1 * testing_multiplier),  # β1: Testing at point 1
            min(0.95, 0.15 * testing_multiplier), # β2: Testing at point 2
            min(0.95, 0.2 * testing_multiplier),  # β3: Testing at point 3
            min(0.95, 0.25 * testing_multiplier)  # β4: Testing at point 4
        ]
        
        self.P = id_probability  # Probability of identifying eligible products
        self.c_e_range = (200, 500)    # Range for effort cost
        self.c_k_range = (500, 1000)   # Range for technology cost
        self.initial_contamination_multiplier = initial_contamination_multiplier
        
        # Initialize farmers with random initial values
        self.farmers = []
        
        # Create risk neutral farmers
        for i in range(self.num_risk_neutral):
            initial_alpha = np.random.uniform(0.4, 0.6)
            initial_technology = np.random.uniform(0.3, 0.7)
            
            farmer = Farmer(
                i, 
                time_steps, 
                alpha=initial_alpha, 
                technology_level=initial_technology,
                risk_preference=Farmer.RISK_NEUTRAL,
                risk_coefficient=1.0
            )
            self.farmers.append(farmer)
        
        # Create risk averse farmers
        for i in range(self.num_risk_neutral, self.num_risk_neutral + self.num_risk_averse):
            initial_alpha = np.random.uniform(0.6, 0.8)  # Higher initial control effort
            initial_technology = np.random.uniform(0.5, 0.8)  # Higher initial technology
            
            farmer = Farmer(
                i, 
                time_steps, 
                alpha=initial_alpha, 
                technology_level=initial_technology,
                risk_preference=Farmer.RISK_AVERSE,
                risk_coefficient=np.random.uniform(0.8, 1.5)  # Random risk coefficient
            )
            self.farmers.append(farmer)
        
        # Create risk loving farmers
        for i in range(self.num_risk_neutral + self.num_risk_averse, num_farmers):
            initial_alpha = np.random.uniform(0.1, 0.4)  # Lower initial control effort
            initial_technology = np.random.uniform(0.2, 0.5)  # Lower initial technology
            
            farmer = Farmer(
                i, 
                time_steps, 
                alpha=initial_alpha, 
                technology_level=initial_technology,
                risk_preference=Farmer.RISK_LOVING,
                risk_coefficient=np.random.uniform(0.5, 1.2)  # Random risk coefficient
            )
            self.farmers.append(farmer)
        
        # Apply initial contamination multiplier
        for farmer in self.farmers:
            farmer.contamination_rate = farmer.calculate_contamination_rate(0) * initial_contamination_multiplier
        
        # Initialize history arrays
        self.mean_alpha_history = np.zeros(time_steps)
        self.mean_contamination_history = np.zeros(time_steps)
        self.mean_cost_history = np.zeros(time_steps)
        
        # Initialize cost tracking arrays
        self.testing_cost_history = np.zeros(time_steps)
        self.penalty_cost_history = np.zeros(time_steps)
        self.effort_cost_history = np.zeros(time_steps)
        self.technology_cost_history = np.zeros(time_steps)
        self.cost_change_history = np.zeros(time_steps-1) if time_steps > 1 else np.zeros(1)
        
        self.mean_alpha_history_by_risk = {
            Farmer.RISK_NEUTRAL: np.zeros(time_steps),
            Farmer.RISK_AVERSE: np.zeros(time_steps),
            Farmer.RISK_LOVING: np.zeros(time_steps)
        }
        self.mean_contamination_history_by_risk = {
            Farmer.RISK_NEUTRAL: np.zeros(time_steps),
            Farmer.RISK_AVERSE: np.zeros(time_steps),
            Farmer.RISK_LOVING: np.zeros(time_steps)
        }
        self.mean_cost_history_by_risk = {
            Farmer.RISK_NEUTRAL: np.zeros(time_steps),
            Farmer.RISK_AVERSE: np.zeros(time_steps),
            Farmer.RISK_LOVING: np.zeros(time_steps)
        }
        self.mean_technology_history_by_risk = {
            Farmer.RISK_NEUTRAL: np.zeros(time_steps),
            Farmer.RISK_AVERSE: np.zeros(time_steps),
            Farmer.RISK_LOVING: np.zeros(time_steps)
        }
        self.testing_cost_history_by_risk = {
            Farmer.RISK_NEUTRAL: np.zeros(time_steps),
            Farmer.RISK_AVERSE: np.zeros(time_steps),
            Farmer.RISK_LOVING: np.zeros(time_steps)
        }
    
    def set_parameters(self, f, beta, P):
        """
        Set model parameters.
        
        Parameters:
        -----------
        f : list of float
            Penalties at test points 1-4 and from illness
        beta : list of float
            Testing probabilities at test points 1-4
        P : float
            Probability that a farmer's eligible products can be identified
        """
        self.f = f
        self.beta = beta
        self.P = P
    
    def enable_differential_testing(self, enable=True):
        """
        Enable or disable the differential testing strategy where risk-loving farmers 
        receive more intensive testing than risk-averse or risk-neutral farmers.
        
        Parameters:
        -----------
        enable : bool, optional
            Whether to enable differential testing (default: True)
        """
        self.differential_testing = enable
        if enable:
            print("Differential testing strategy enabled: Risk-loving farmers will be tested more intensively.")
        else:
            print("Uniform testing strategy enabled: All farmers receive equal testing rates.")
    
    def set_test_identification_probability(self, prob):
        """
        Set the identification probability for tests (accuracy of tests).
        Lower values simulate less accurate testing technology.
        
        Parameters:
        -----------
        prob : float
            Probability that a test correctly identifies contamination (0-1)
        """
        self.test_identification_prob = max(0.1, min(1.0, prob))
        print(f"Test identification probability set to {self.test_identification_prob}")
    
    def analyze_testing_strategies(self, output_dir='results/testing_strategies'):
        """
        Analyze the effect of different testing strategies on contamination rates and costs.
        Compares uniform testing versus targeted testing of risk-loving farmers.
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save the results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Store current parameters to restore later
        current_beta = self.beta.copy()
        current_differential_testing = hasattr(self, 'differential_testing') and self.differential_testing
        
        testing_rates = [0.5, 1.0, 1.5, 2.0]  # Multipliers for testing rates
        strategies = ['uniform', 'differential']
        
        results = {
            'testing_rate': [],
            'strategy': [],
            'mean_contamination': [],
            'mean_cost': [],
            'risk_neutral_contamination': [],
            'risk_averse_contamination': [],
            'risk_loving_contamination': [],
            'risk_neutral_cost': [],
            'risk_averse_cost': [],
            'risk_loving_cost': []
        }
        
        # Run simulations for each testing rate and strategy
        for rate in testing_rates:
            for strategy in strategies:
                # Set testing parameters
                self.beta = [
                    min(0.95, 0.1 * rate),
                    min(0.95, 0.15 * rate),
                    min(0.95, 0.2 * rate),
                    min(0.95, 0.25 * rate)
                ]
                
                # Set testing strategy
                self.enable_differential_testing(strategy == 'differential')
                
                # Reset and run simulation
                # Reset farmers' parameters
                for farmer in self.farmers:
                    if farmer.risk_preference == Farmer.RISK_NEUTRAL:
                        farmer.alpha = np.random.uniform(0.4, 0.6)
                        farmer.technology_level = np.random.uniform(0.3, 0.7)
                    elif farmer.risk_preference == Farmer.RISK_AVERSE:
                        farmer.alpha = np.random.uniform(0.6, 0.8)
                        farmer.technology_level = np.random.uniform(0.5, 0.8)
                    elif farmer.risk_preference == Farmer.RISK_LOVING:
                        farmer.alpha = np.random.uniform(0.1, 0.4)
                        farmer.technology_level = np.random.uniform(0.2, 0.5)
                
                # Run simulation
                print(f"\nRunning simulation with {strategy} testing strategy at rate {rate}...")
                self.run_simulation()
                
                # Store results
                results['testing_rate'].append(rate)
                results['strategy'].append(strategy)
                results['mean_contamination'].append(self.mean_contamination_history[-1])
                results['mean_cost'].append(self.mean_cost_history[-1])
                
                for risk_type in [Farmer.RISK_NEUTRAL, Farmer.RISK_AVERSE, Farmer.RISK_LOVING]:
                    type_name = Farmer.RISK_TYPE_NAMES[risk_type].lower().replace(' ', '_')
                    results[f'{type_name}_contamination'].append(self.mean_contamination_history_by_risk[risk_type][-1])
                    results[f'{type_name}_cost'].append(self.mean_cost_history_by_risk[risk_type][-1])
                
                # Save results for this configuration
                sub_dir = os.path.join(output_dir, f"{strategy}_rate_{rate}")
                os.makedirs(sub_dir, exist_ok=True)
                self.save_results_to_file(sub_dir)
                self.plot_results(sub_dir)
        
        # Restore original parameters
        self.beta = current_beta
        if hasattr(self, 'differential_testing'):
            self.differential_testing = current_differential_testing
        
        # Create comparative plots
        self._plot_testing_strategy_comparison(results, output_dir)
        
        # Return results for further analysis
        return results
    
    def _plot_testing_strategy_comparison(self, results, output_dir):
        """
        Plot the comparative results of different testing strategies.
        
        Parameters:
        -----------
        results : dict
            Dictionary containing the results of the testing strategy analysis
        output_dir : str
            Directory to save the plots
        """
        # Convert results to DataFrame for easier plotting
        results_df = pd.DataFrame({
            'testing_rate': results['testing_rate'],
            'strategy': results['strategy'],
            'mean_contamination': results['mean_contamination'],
            'mean_cost': results['mean_cost'],
            'risk_neutral_contamination': results['risk_neutral_contamination'],
            'risk_averse_contamination': results['risk_averse_contamination'],
            'risk_loving_contamination': results['risk_loving_contamination'],
            'risk_neutral_cost': results['risk_neutral_cost'],
            'risk_averse_cost': results['risk_averse_cost'],
            'risk_loving_cost': results['risk_loving_cost']
        })
        
        # Plot mean contamination by testing rate and strategy
        plt.figure(figsize=(10, 6))
        for strategy in ['uniform', 'differential']:
            strategy_data = results_df[results_df['strategy'] == strategy]
            plt.plot(strategy_data['testing_rate'], strategy_data['mean_contamination'], 
                     'o-', label=f"{strategy.capitalize()} Testing")
        
        plt.xlabel('Testing Rate Multiplier')
        plt.ylabel('Mean Contamination Rate')
        plt.title('Effect of Testing Strategy on Mean Contamination Rate')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'strategy_contamination_comparison.png'))
        plt.close()
        
        # Plot mean cost by testing rate and strategy
        plt.figure(figsize=(10, 6))
        for strategy in ['uniform', 'differential']:
            strategy_data = results_df[results_df['strategy'] == strategy]
            plt.plot(strategy_data['testing_rate'], strategy_data['mean_cost'], 
                     'o-', label=f"{strategy.capitalize()} Testing")
        
        plt.xlabel('Testing Rate Multiplier')
        plt.ylabel('Mean Cost')
        plt.title('Effect of Testing Strategy on Mean Cost')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'strategy_cost_comparison.png'))
        plt.close()
        
        # Plot contamination by risk type for each strategy
        testing_rates = sorted(set(results_df['testing_rate']))
        strategies = ['uniform', 'differential']
        
        for rate in testing_rates:
            plt.figure(figsize=(12, 8))
            
            rate_data = results_df[results_df['testing_rate'] == rate]
            
            x = np.arange(len(strategies))
            width = 0.25
            
            risk_types = ['risk_neutral', 'risk_averse', 'risk_loving']
            colors = ['blue', 'green', 'red']
            
            for i, risk_type in enumerate(risk_types):
                contamination_values = [
                    rate_data[rate_data['strategy'] == strategy][f'{risk_type}_contamination'].values[0]
                    for strategy in strategies
                ]
                plt.bar(x + i*width - width, contamination_values, width, 
                        label=f"{risk_type.replace('_', ' ').title()}", color=colors[i])
            
            plt.xlabel('Testing Strategy')
            plt.ylabel('Contamination Rate')
            plt.title(f'Contamination Rate by Risk Type (Testing Rate = {rate})')
            plt.xticks(x, [s.capitalize() for s in strategies])
            plt.legend()
            plt.grid(True, axis='y')
            plt.savefig(os.path.join(output_dir, f'risk_type_contamination_rate_{rate}.png'))
            plt.close()
    
    def run_simulation(self):
        """
        Run the simulation for the specified number of time steps.
        """
        # Calculate regional benchmark for contamination detection (h in Eq. 6)
        # Initially set to 1 detection per region, but will adjust with data
        regional_benchmark = 1
        
        # Random innovation probability (affects D₃)
        innovation_prob = 0.05
        
        for t in range(self.time_steps):
            alphas = []
            contamination_rates = []
            costs = []
            testing_costs = []
            penalty_costs = []
            effort_costs = []
            technology_costs = []
            
            # Track metrics by risk preference
            alphas_by_risk = {
                Farmer.RISK_NEUTRAL: [],
                Farmer.RISK_AVERSE: [],
                Farmer.RISK_LOVING: []
            }
            contamination_by_risk = {
                Farmer.RISK_NEUTRAL: [],
                Farmer.RISK_AVERSE: [],
                Farmer.RISK_LOVING: []
            }
            costs_by_risk = {
                Farmer.RISK_NEUTRAL: [],
                Farmer.RISK_AVERSE: [],
                Farmer.RISK_LOVING: []
            }
            technology_by_risk = {
                Farmer.RISK_NEUTRAL: [],
                Farmer.RISK_AVERSE: [],
                Farmer.RISK_LOVING: []
            }
            testing_costs_by_risk = {
                Farmer.RISK_NEUTRAL: [],
                Farmer.RISK_AVERSE: [],
                Farmer.RISK_LOVING: []
            }
            
            # First pass: calculate contamination rates and simulate testing for all farmers
            total_tests_conducted = 0
            tests_by_risk_type = {
                Farmer.RISK_NEUTRAL: 0,
                Farmer.RISK_AVERSE: 0,
                Farmer.RISK_LOVING: 0
            }
            
            for farmer in self.farmers:
                # Calculate contamination probability based on current alpha and technology
                prev_contamination_rate = farmer.contamination_rate
                if prev_contamination_rate is None:
                    # First time step, calculate initial rate
                    prev_contamination_rate = farmer.calculate_contamination_rate(0)
                
                # Simulate testing and contamination detection
                # 1. Determine if the farmer's produce will be tested at any point
                beta1, beta2, beta3, beta4 = self.beta
                
                # Apply different testing rates based on risk preference if differential testing is enabled
                farmer_type_adjustment = 1.0
                if hasattr(self, 'differential_testing') and self.differential_testing:
                    if farmer.risk_preference == Farmer.RISK_LOVING:
                        # Higher testing for risk-loving farmers
                        farmer_type_adjustment = 1.5
                    elif farmer.risk_preference == Farmer.RISK_AVERSE:
                        # Lower testing for risk-averse farmers
                        farmer_type_adjustment = 0.7
                
                # Apply the adjustment
                adjusted_beta1 = min(0.95, beta1 * farmer_type_adjustment)
                adjusted_beta2 = min(0.95, beta2 * farmer_type_adjustment)
                adjusted_beta3 = min(0.95, beta3 * farmer_type_adjustment)
                adjusted_beta4 = min(0.95, beta4 * farmer_type_adjustment)
                
                # Calculate testing probability at each stage
                test_at_stage1 = adjusted_beta1
                test_at_stage2 = (1 - adjusted_beta1) * adjusted_beta2
                test_at_stage3 = (1 - adjusted_beta1) * (1 - adjusted_beta2) * adjusted_beta3
                test_at_stage4 = (1 - adjusted_beta1) * (1 - adjusted_beta2) * (1 - adjusted_beta3) * adjusted_beta4
                
                # Calculate expected number of tests for this farmer
                expected_tests = test_at_stage1 + test_at_stage2 + test_at_stage3 + test_at_stage4
                
                # Track total tests conducted
                farmer_tests = np.random.binomial(1, expected_tests)
                total_tests_conducted += farmer_tests
                tests_by_risk_type[farmer.risk_preference] += farmer_tests
                
                # Determine if farmer was tested at any point
                if farmer_tests > 0 or np.random.random() < expected_tests:
                    farmer.was_tested = True
                else:
                    farmer.was_tested = False
                
                # Determine if contamination is actually present
                is_contaminated = np.random.random() < prev_contamination_rate
                
                # Simulate if contamination is detected (D₂)
                detection_probability = 0
                if is_contaminated:
                    # Probability of detection at any of the four test points
                    test_detection_prob = 1 - (1-adjusted_beta1)*(1-adjusted_beta2)*(1-adjusted_beta3)*(1-adjusted_beta4)
                    
                    # Add probability of illness identification through tracing
                    traceback_detection_prob = (1-adjusted_beta1)*(1-adjusted_beta2)*(1-adjusted_beta3)*(1-adjusted_beta4) * self.P
                    
                    # Combined detection probability
                    detection_probability = test_detection_prob + traceback_detection_prob - (test_detection_prob * traceback_detection_prob)
                    
                    # Account for imperfect testing detection
                    if hasattr(self, 'test_identification_prob'):
                        detection_probability *= self.test_identification_prob
                    
                    # Check if detected
                    if np.random.random() < detection_probability:
                        farmer.contamination_detected = True
                    else:
                        farmer.contamination_detected = False
                else:
                    farmer.contamination_detected = False
                
                # Simulate technology innovation adoption (D₃)
                if np.random.random() < innovation_prob:
                    farmer.tech_innovation_adopted = True
                else:
                    farmer.tech_innovation_adopted = False
            
            # Calculate testing cost for this time step
            time_step_testing_cost = total_tests_conducted * self.testing_cost_per_sample
            self.testing_cost_history[t] = time_step_testing_cost
            
            # Calculate testing costs by risk type
            for risk_type in [Farmer.RISK_NEUTRAL, Farmer.RISK_AVERSE, Farmer.RISK_LOVING]:
                risk_type_testing_cost = tests_by_risk_type[risk_type] * self.testing_cost_per_sample
                self.testing_cost_history_by_risk[risk_type][t] = risk_type_testing_cost
            
            # Second pass: update all farmers based on the testing results
            detection_count = sum(1 for f in self.farmers if f.contamination_detected)
            if t > 5:  # After some burn-in period
                # Update regional benchmark based on recent detection rate
                regional_benchmark = max(1, detection_count / 2)  # At least 1 detection as benchmark
            
            # Calculate neighbors for each farmer - all farmers know all others in this simplified model
            for farmer in self.farmers:
                # Get all other farmers as neighbors (simplified network)
                neighbors = [f for f in self.farmers if f.id != farmer.id]
                
                # Random costs for effort and technology for each farmer
                c_e = np.random.uniform(self.c_e_range[0], self.c_e_range[1])
                c_k = np.random.uniform(self.c_k_range[0], self.c_k_range[1])
                
                # Update the farmer's risk control effort and technology
                alpha, contamination_rate, cost = farmer.update(
                    t, self.f, self.beta, self.P, c_e, c_k, neighbors, regional_benchmark
                )
                
                # Collect results
                alphas.append(alpha)
                contamination_rates.append(contamination_rate)
                if not np.isinf(cost) and not np.isnan(cost):
                    costs.append(cost)
                
                # Collect results by risk preference
                alphas_by_risk[farmer.risk_preference].append(alpha)
                contamination_by_risk[farmer.risk_preference].append(contamination_rate)
                if not np.isinf(cost) and not np.isnan(cost):
                    costs_by_risk[farmer.risk_preference].append(cost)
                technology_by_risk[farmer.risk_preference].append(farmer.technology_level)
                
                # Track cost components
                effort_cost = c_e * alpha
                tech_cost = c_k * farmer.technology_level
                effort_costs.append(effort_cost)
                technology_costs.append(tech_cost)
                
                # Calculate expected penalty cost (part of the total cost)
                # This is a simplified approximation since we don't track actual penalties paid
                expected_penalty = contamination_rate * sum(self.f) * 0.1  # Rough estimate of expected penalty
                penalty_costs.append(expected_penalty)
            
            # Store aggregate results for this time step
            self.mean_alpha_history[t] = np.mean(alphas)
            self.mean_contamination_history[t] = np.mean(contamination_rates)
            
            # Filter out invalid costs
            valid_costs = [c for c in costs if not np.isinf(c) and not np.isnan(c)]
            self.mean_cost_history[t] = np.mean(valid_costs) if valid_costs else float('nan')
            
            # Store results by risk preference
            for risk_type in [Farmer.RISK_NEUTRAL, Farmer.RISK_AVERSE, Farmer.RISK_LOVING]:
                if alphas_by_risk[risk_type]:
                    self.mean_alpha_history_by_risk[risk_type][t] = np.mean(alphas_by_risk[risk_type])
                    self.mean_contamination_history_by_risk[risk_type][t] = np.mean(contamination_by_risk[risk_type])
                    
                    # Filter out invalid costs
                    valid_costs = [c for c in costs_by_risk[risk_type] if not np.isinf(c) and not np.isnan(c)]
                    self.mean_cost_history_by_risk[risk_type][t] = np.mean(valid_costs) if valid_costs else float('nan')
                    
                    self.mean_technology_history_by_risk[risk_type][t] = np.mean(technology_by_risk[risk_type])
            
            # Print current status
            contamination_pct = self.mean_contamination_history[t] * 100
            detections_pct = (detection_count / len(self.farmers)) * 100
            print(f"Time step {t}: Mean α = {self.mean_alpha_history[t]:.4f}, Mean contamination = {contamination_pct:.2f}%, " + 
                  f"Detected = {detection_count} ({detections_pct:.2f}%), Mean cost = {self.mean_cost_history[t]:.2f}")

    def save_results_to_file(self, output_dir):
        """
        Save simulation results to files in the specified directory.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save results in
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model parameters
        params_file = os.path.join(output_dir, 'parameters.txt')
        with open(params_file, 'w') as f:
            f.write(f"Number of farmers: {self.num_farmers}\n")
            f.write(f"- Risk neutral: {self.num_risk_neutral}\n")
            f.write(f"- Risk averse: {self.num_risk_averse}\n")
            f.write(f"- Risk loving: {self.num_risk_loving}\n")
            f.write(f"Time steps: {self.time_steps}\n")
            f.write(f"Penalty values: {self.f}\n")
            f.write(f"Testing probabilities: {self.beta}\n")
            f.write(f"Identification probability: {self.P}\n")
            f.write(f"Effort cost range: {self.c_e_range}\n")
            f.write(f"Technology cost range: {self.c_k_range}\n")
        
        # Save aggregate results
        results_file = os.path.join(output_dir, 'results.csv')
        with open(results_file, 'w') as f:
            f.write("time_step,mean_alpha,mean_contamination,mean_cost\n")
            for t in range(self.time_steps):
                f.write(f"{t},{self.mean_alpha_history[t]},{self.mean_contamination_history[t]},{self.mean_cost_history[t]}\n")
        
        # Save results by risk preference
        risk_results_file = os.path.join(output_dir, 'results_by_risk.csv')
        with open(risk_results_file, 'w') as f:
            f.write("time_step,risk_type,mean_alpha,mean_contamination,mean_cost,mean_technology\n")
            for t in range(self.time_steps):
                for risk_type in [Farmer.RISK_NEUTRAL, Farmer.RISK_AVERSE, Farmer.RISK_LOVING]:
                    risk_name = Farmer.RISK_TYPE_NAMES[risk_type].replace(" ", "_").lower()
                    f.write(f"{t},{risk_name},{self.mean_alpha_history_by_risk[risk_type][t]},"
                           f"{self.mean_contamination_history_by_risk[risk_type][t]},"
                           f"{self.mean_cost_history_by_risk[risk_type][t]},"
                           f"{self.mean_technology_history_by_risk[risk_type][t]}\n")
        
        # Save individual farmer data
        farmer_file = os.path.join(output_dir, 'farmer_data.csv')
        with open(farmer_file, 'w') as f:
            f.write("farmer_id,risk_type,time_step,alpha,contamination,cost,technology\n")
            for farmer in self.farmers:
                risk_name = Farmer.RISK_TYPE_NAMES[farmer.risk_preference].replace(" ", "_").lower()
                for t in range(self.time_steps):
                    f.write(f"{farmer.id},{risk_name},{t},{farmer.alpha_history[t]},"
                           f"{farmer.contamination_history[t]},{farmer.cost_history[t]},"
                           f"{farmer.technology_history[t]}\n")

    def plot_results(self, output_dir='results'):
        """
        Plot the simulation results.
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save results
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        time_steps_range = range(self.time_steps)
        
        # Create a figure with 3 subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot mean alpha over time
        axs[0].plot(time_steps_range, self.mean_alpha_history, label='Overall')
        for risk_type in [Farmer.RISK_NEUTRAL, Farmer.RISK_AVERSE, Farmer.RISK_LOVING]:
            axs[0].plot(time_steps_range, self.mean_alpha_history_by_risk[risk_type], 
                       label=Farmer.RISK_TYPE_NAMES[risk_type])
        axs[0].set_xlabel('Time Step')
        axs[0].set_ylabel('Mean Risk Control Effort (α)')
        axs[0].set_title('Mean Risk Control Effort over Time by Risk Preference')
        axs[0].grid(True)
        axs[0].legend()
        
        # Plot mean contamination rate over time
        axs[1].plot(time_steps_range, self.mean_contamination_history, label='Overall')
        for risk_type in [Farmer.RISK_NEUTRAL, Farmer.RISK_AVERSE, Farmer.RISK_LOVING]:
            axs[1].plot(time_steps_range, self.mean_contamination_history_by_risk[risk_type], 
                       label=Farmer.RISK_TYPE_NAMES[risk_type])
        axs[1].set_xlabel('Time Step')
        axs[1].set_ylabel('Mean Contamination Rate (σ)')
        axs[1].set_title('Mean Contamination Rate over Time by Risk Preference')
        axs[1].grid(True)
        axs[1].legend()
        
        # Plot mean cost over time
        axs[2].plot(time_steps_range, self.mean_cost_history, label='Overall')
        for risk_type in [Farmer.RISK_NEUTRAL, Farmer.RISK_AVERSE, Farmer.RISK_LOVING]:
            axs[2].plot(time_steps_range, self.mean_cost_history_by_risk[risk_type], 
                       label=Farmer.RISK_TYPE_NAMES[risk_type])
        axs[2].set_xlabel('Time Step')
        axs[2].set_ylabel('Mean Cost')
        axs[2].set_title('Mean Cost over Time by Risk Preference')
        axs[2].grid(True)
        axs[2].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'simulation_results.png'))
        plt.close()
        
        # Plot technology vs contamination by risk type
        plt.figure(figsize=(12, 8))
        for risk_type, name in Farmer.RISK_TYPE_NAMES.items():
            plt.plot(self.mean_technology_history_by_risk[risk_type], 
                    self.mean_contamination_history_by_risk[risk_type], 
                    'o-', label=name)
        plt.xlabel('Mean Technology Level (k)')
        plt.ylabel('Mean Contamination Rate (σ)')
        plt.title('Technology Level vs. Contamination Rate by Risk Preference')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'technology_vs_contamination.png'))
        plt.close()
        
        # Plot technology over time by risk type
        plt.figure(figsize=(12, 8))
        for risk_type, name in Farmer.RISK_TYPE_NAMES.items():
            plt.plot(time_steps_range, self.mean_technology_history_by_risk[risk_type], 
                    label=name)
        plt.xlabel('Time Step')
        plt.ylabel('Mean Technology Level (k)')
        plt.title('Technology Level over Time by Risk Preference')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'technology_over_time.png'))
        plt.close()
        
        # Plot individual farmer trajectories grouped by risk preference
        for risk_type, name in Farmer.RISK_TYPE_NAMES.items():
            plt.figure(figsize=(12, 8))
            
            farmers_of_type = [f for f in self.farmers if f.risk_preference == risk_type]
            sample_size = min(10, len(farmers_of_type))
            
            for i in range(sample_size):
                plt.plot(time_steps_range, farmers_of_type[i].alpha_history, 
                        label=f'Farmer {farmers_of_type[i].id}')
            
            plt.xlabel('Time Step')
            plt.ylabel('Risk Control Effort (α)')
            plt.title(f'Individual {name} Farmer Risk Control Effort Trajectories')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'individual_trajectories_{risk_type}.png'))
            plt.close()
        
        # Additional plot: distribution of final alpha values by risk type
        plt.figure(figsize=(12, 8))
        for risk_type, name in Farmer.RISK_TYPE_NAMES.items():
            final_alphas = [farmer.alpha_history[-1] for farmer in self.farmers 
                          if farmer.risk_preference == risk_type]
            if final_alphas:  # Check if list is not empty
                plt.hist(final_alphas, bins=15, alpha=0.6, label=name)
        
        plt.xlabel('Final Risk Control Effort (α)')
        plt.ylabel('Number of Farmers')
        plt.title('Distribution of Final Risk Control Effort by Risk Preference')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'alpha_distribution_by_risk.png'))
        plt.close()

    def plot_mathematical_equations(self, output_dir='results/equations'):
        """
        Create visual representations of the mathematical equations used in the model.
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save the equation visualization plots
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Plot 1: Contamination Rate Function (Equation 3)
        # σ_j^t = e^(-c_j^t * k_j)
        effort_values = np.linspace(0.05, 0.95, 100)
        tech_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
        
        plt.figure(figsize=(10, 6))
        for tech in tech_levels:
            contamination_rates = [np.exp(-effort * tech) for effort in effort_values]
            plt.plot(effort_values, contamination_rates, label=f'Technology Level (k) = {tech}')
        
        plt.xlabel('Risk Control Effort (c)')
        plt.ylabel('Contamination Rate (σ)')
        plt.title('Equation 3: Contamination Rate as a Function of Effort and Technology')
        plt.grid(True)
        plt.legend()
        # Add equation as text
        plt.text(0.1, 0.2, r'$\sigma_j^t = e^{-c_j^t \cdot k_j}$', fontsize=14, 
                 bbox=dict(facecolor='white', alpha=0.8))
        plt.savefig(os.path.join(output_dir, 'equation3_contamination_rate.png'))
        plt.close()
        
        # Plot 2: Effect of Risk Preference on Cost Perception
        alpha_values = np.linspace(0.05, 0.95, 100)
        base_cost = 1000  # Base cost
        
        risk_coef = 1.0  # Risk coefficient
        risk_averse_costs = [base_cost * (1 + risk_coef * 0.5) for _ in alpha_values]
        risk_neutral_costs = [base_cost for _ in alpha_values]
        risk_loving_costs = [base_cost * (1 - risk_coef * 0.3) for _ in alpha_values]
        
        plt.figure(figsize=(10, 6))
        plt.plot(alpha_values, risk_averse_costs, label='Risk Averse')
        plt.plot(alpha_values, risk_neutral_costs, label='Risk Neutral')
        plt.plot(alpha_values, risk_loving_costs, label='Risk Loving')
        plt.xlabel('Risk Control Effort (α)')
        plt.ylabel('Perceived Cost')
        plt.title('Effect of Risk Preference on Cost Perception')
        plt.grid(True)
        plt.legend()
        # Add equation as text
        plt.text(0.1, 1300, r'Risk Averse: Cost × (1 + risk_coef × 0.5)', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8))
        plt.text(0.1, 1200, r'Risk Neutral: Cost', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8))
        plt.text(0.1, 1100, r'Risk Loving: Cost × (1 - risk_coef × 0.3)', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8))
        plt.savefig(os.path.join(output_dir, 'risk_preference_cost_effect.png'))
        plt.close()
        
        # Plot 3: Cost Function Components (Equation 4)
        alpha_values = np.linspace(0.05, 0.95, 100)
        
        # Define sample parameters
        f1, f2, f3, f4, f5 = 100, 300, 600, 1000, 5000  # Penalties
        beta1, beta2, beta3, beta4 = 0.1, 0.15, 0.2, 0.25  # Testing probabilities
        P = 0.5  # Identification probability
        c_e, c_k = 300, 700  # Effort and technology costs
        k = 0.5  # Technology level
        
        # Calculate cost components
        direct_costs = [c_e * alpha + c_k * k for alpha in alpha_values]
        
        # Expected penalties
        penalties = []
        for alpha in alpha_values:
            contamination = np.exp(-alpha * k)  # Calculate contamination rate
            # Expected penalty calculation from Equation 4
            expected_penalty = (
                f1 * beta1 * contamination +
                f2 * (1 - beta1) * beta2 * contamination +
                f3 * (1 - beta1) * (1 - beta2) * beta3 * contamination +
                f4 * (1 - beta1) * (1 - beta2) * (1 - beta3) * beta4 * contamination +
                f5 * (1 - beta1) * (1 - beta2) * (1 - beta3) * (1 - beta4) * P * contamination
            )
            # Normalize by pass probability
            pass_probability = (1 - beta1) * (1 - beta2) * (1 - beta3) * (1 - beta4)
            expected_penalty = expected_penalty / pass_probability if pass_probability > 0 else float('inf')
            penalties.append(expected_penalty)
        
        # Total costs
        total_costs = [d + p for d, p in zip(direct_costs, penalties)]
        
        plt.figure(figsize=(10, 6))
        plt.plot(alpha_values, direct_costs, label='Direct Costs (Effort + Technology)')
        plt.plot(alpha_values, penalties, label='Expected Penalties')
        plt.plot(alpha_values, total_costs, label='Total Cost')
        plt.xlabel('Risk Control Effort (α)')
        plt.ylabel('Cost')
        plt.title('Equation 4: Cost Function Components')
        plt.grid(True)
        plt.legend()
        # Add simplified equation as text
        plt.text(0.1, 6000, r'$Total Cost = Direct Costs + \frac{Expected Penalties}{Pass Probability}$', 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        plt.savefig(os.path.join(output_dir, 'equation4_cost_function.png'))
        plt.close()
        
        # Plot 4: Optimal Contamination Rate (Equation 5)
        alpha_values = np.linspace(0.05, 0.95, 100)
        
        # Calculate costs for different effort levels
        costs = []
        for alpha in alpha_values:
            contamination = np.exp(-alpha * k)
            # Cost calculation from Equation 4
            expected_penalty = (
                f1 * beta1 * contamination +
                f2 * (1 - beta1) * beta2 * contamination +
                f3 * (1 - beta1) * (1 - beta2) * beta3 * contamination +
                f4 * (1 - beta1) * (1 - beta2) * (1 - beta3) * beta4 * contamination +
                f5 * (1 - beta1) * (1 - beta2) * (1 - beta3) * (1 - beta4) * P * contamination
            )
            pass_probability = (1 - beta1) * (1 - beta2) * (1 - beta3) * (1 - beta4)
            penalty_cost = expected_penalty / pass_probability if pass_probability > 0 else float('inf')
            direct_cost = c_e * alpha + c_k * k
            costs.append(direct_cost + penalty_cost)
        
        # Find the optimal alpha (minimum cost)
        valid_costs = [c if not np.isinf(c) and not np.isnan(c) else float('inf') for c in costs]
        min_cost_idx = np.argmin(valid_costs)
        optimal_alpha = alpha_values[min_cost_idx]
        min_cost = valid_costs[min_cost_idx]
        
        plt.figure(figsize=(10, 6))
        plt.plot(alpha_values, costs)
        plt.axvline(x=optimal_alpha, color='r', linestyle='--', 
                  label=f'Optimal α = {optimal_alpha:.2f}')
        plt.scatter([optimal_alpha], [min_cost], color='red', s=100, zorder=5)
        plt.xlabel('Risk Control Effort (α)')
        plt.ylabel('Total Cost')
        plt.title('Equation 5: Finding Optimal Risk Control Effort')
        plt.grid(True)
        plt.legend()
        # Add equation as text
        plt.text(0.6, min_cost * 0.8, 
                r'$\alpha^* = \arg\min_{\alpha} \: Cost(\alpha)$', 
                fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
        plt.savefig(os.path.join(output_dir, 'equation5_optimal_effort.png'))
        plt.close()
        
        # Plot 5: Risk Preference Effect on Optimal Alpha
        # Calculate optimal alpha for different risk preferences
        risk_preferences = [Farmer.RISK_NEUTRAL, Farmer.RISK_AVERSE, Farmer.RISK_LOVING]
        risk_coefs = [1.0, 1.5, 0.7]  # Coefficients for neutral, averse, loving
        
        plt.figure(figsize=(12, 8))
        
        for i, (risk_pref, risk_coef) in enumerate(zip(risk_preferences, risk_coefs)):
            modified_costs = []
            
            for alpha in alpha_values:
                contamination = np.exp(-alpha * k)
                
                # Adjust penalties based on risk preference
                if risk_pref == Farmer.RISK_AVERSE:
                    penalty_factor = 1 + risk_coef * 0.5
                    modified_c_e = c_e * 0.8
                elif risk_pref == Farmer.RISK_LOVING:
                    penalty_factor = 1 - risk_coef * 0.3
                    modified_c_e = c_e * 1.5
                else:
                    penalty_factor = 1.0
                    modified_c_e = c_e
                
                # Apply risk preference adjustment to penalties
                adjusted_f1 = f1 * penalty_factor
                adjusted_f2 = f2 * penalty_factor
                adjusted_f3 = f3 * penalty_factor
                adjusted_f4 = f4 * penalty_factor
                adjusted_f5 = f5 * penalty_factor
                
                # Calculate expected penalty with adjusted values
                expected_penalty = (
                    adjusted_f1 * beta1 * contamination +
                    adjusted_f2 * (1 - beta1) * beta2 * contamination +
                    adjusted_f3 * (1 - beta1) * (1 - beta2) * beta3 * contamination +
                    adjusted_f4 * (1 - beta1) * (1 - beta2) * (1 - beta3) * beta4 * contamination +
                    adjusted_f5 * (1 - beta1) * (1 - beta2) * (1 - beta3) * (1 - beta4) * P * contamination
                )
                pass_probability = (1 - beta1) * (1 - beta2) * (1 - beta3) * (1 - beta4)
                penalty_cost = expected_penalty / pass_probability if pass_probability > 0 else float('inf')
                direct_cost = modified_c_e * alpha + c_k * k
                modified_costs.append(direct_cost + penalty_cost)
            
            # Find optimal alpha for this risk preference
            valid_costs = [c if not np.isinf(c) and not np.isnan(c) else float('inf') for c in modified_costs]
            min_cost_idx = np.argmin(valid_costs)
            optimal_alpha = alpha_values[min_cost_idx]
            min_cost = valid_costs[min_cost_idx]
            
            # Plot cost curve
            color = ['blue', 'green', 'red'][i]
            plt.plot(alpha_values, modified_costs, color=color, 
                    label=f"{Farmer.RISK_TYPE_NAMES[risk_pref]}")
            
            # Mark optimal point
            plt.axvline(x=optimal_alpha, color=color, linestyle='--')
            plt.scatter([optimal_alpha], [min_cost], color=color, s=100, zorder=5)
            plt.text(optimal_alpha + 0.02, min_cost, 
                    f'α* = {optimal_alpha:.2f}', color=color, fontsize=12)
        
        plt.xlabel('Risk Control Effort (α)')
        plt.ylabel('Total Cost')
        plt.title('Effect of Risk Preference on Optimal Risk Control Effort')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'risk_preference_optimal_effort.png'))
        plt.close()
        
        print(f"Mathematical equation visualizations saved to '{output_dir}'")


# Example of how to use the model
if __name__ == "__main__":
    # Create a model with 100 farmers and 50 time steps
    model = FarmerRiskControlModel(num_farmers=100, time_steps=50, seed=42)
    
    # Set model parameters
    f = [100, 200, 300, 400, 500]  # Penalties (f1, f2, f3, f4, f5)
    beta = [0.1, 0.2, 0.3, 0.4]  # Testing probabilities (β1, β2, β3, β4)
    P = 0.5  # Probability of identifying eligible products
    
    model.set_parameters(f, beta, P)
    
    # Run the simulation
    model.run_simulation()
    
    # Save the results to file
    model.save_results_to_file('results')
    
    # Plot the results
    model.plot_results()
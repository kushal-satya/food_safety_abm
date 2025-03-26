import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class Farmer:
    """
    Agent class representing a farmer in the food safety ABM model.
    This implements the risk control behavior model as specified in Model 3.
    """
    def __init__(self, farmer_id, time_steps, alpha=0.5, technology_level=0.5):
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
        """
        self.id = farmer_id
        self.alpha = alpha  # Initial risk control effort
        self.technology_level = technology_level  # k parameter in the model
        self.contamination_rate = None  # Will be calculated using the exponential function
        
        # Initialize arrays to store history
        self.time_steps = time_steps
        self.alpha_history = np.zeros(time_steps)
        self.contamination_history = np.zeros(time_steps)
        self.cost_history = np.zeros(time_steps)
        
        # Cost components
        self.c_e = 0  # Effort cost
        self.c_k = 0  # Technology adaptation cost
        self.penalties = [0, 0, 0, 0, 0]  # f1, f2, f3, f4, f5
        self.testing_regimes = [0, 0, 0, 0]  # β1, β2, β3, β4
        
    def calculate_contamination_rate(self, t):
        """
        Calculate contamination rate using the exponential function from Eq. (3)
        σ_j^t = e^(-c_j^t * k_j)
        """
        c_j_t = self.alpha
        k_j = self.technology_level
        return np.exp(-c_j_t * k_j)
    
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
        
        # Calculate numerator
        numerator = f1 * beta1 + f2 * (1 - beta1) * beta2 + \
                   f3 * (1 - beta1) * (1 - beta2) * beta3 + \
                   f4 * (1 - beta1) * (1 - beta2) * (1 - beta3) * beta4 + \
                   f5 * (1 - beta1) * (1 - beta2) * (1 - beta3) * (1 - beta4) * P
        
        # Calculate denominator
        denominator = (c_e + c_k) * (1 - beta1) * (1 - beta2) * (1 - beta3) * (1 - beta4)
        
        # Store values for later reference
        self.c_e = c_e
        self.c_k = c_k
        self.penalties = [f1, f2, f3, f4, f5]
        self.testing_regimes = [beta1, beta2, beta3, beta4]
        
        if denominator == 0:
            return float('inf')
        
        return numerator / denominator
    
    def find_optimal_contamination_rate(self, f, beta, P, c_e, c_k):
        """
        Find the optimal contamination rate based on Eq. (5)
        α = (f1*β1 + f2*(1-β1)*β2 + f3*(1-β1)*(1-β2)*β3 + f4*(1-β1)*(1-β2)*(1-β3)*β4 + f5*(1-β1)*(1-β2)*(1-β3)*(1-β4)*P) / 
            ((c_e+c_k)*(1-β1)*(1-β2)*(1-β3)*(1-β4))
        """
        # We'll use the calculate_cost method and scipy's minimize to find the optimal alpha
        def objective(alpha_array):
            self.alpha = alpha_array[0]
            return self.calculate_cost(f, beta, P, c_e, c_k)
        
        # Bounds for alpha (between 0 and 1)
        bounds = [(0, 1)]
        
        # Initial guess
        x0 = [self.alpha]
        
        # Find the minimum cost and corresponding alpha
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            self.alpha = result.x[0]
        else:
            print(f"Optimization failed for farmer {self.id}: {result.message}")
        
        return self.alpha
    
    def update(self, t, f, beta, P, c_e, c_k, neighbors=None):
        """
        Update the farmer's risk control effort and contamination rate for time step t
        
        Parameters:
        -----------
        t : int
            Current time step
        f, beta, P, c_e, c_k : model parameters as described in calculate_cost
        neighbors : list of Farmer, optional
            List of neighboring farmers that may influence this farmer's decision
        """
        # Find optimal alpha
        self.find_optimal_contamination_rate(f, beta, P, c_e, c_k)
        
        # Calculate contamination rate
        self.contamination_rate = self.calculate_contamination_rate(t)
        
        # Calculate cost
        cost = self.calculate_cost(f, beta, P, c_e, c_k)
        
        # Store history
        self.alpha_history[t] = self.alpha
        self.contamination_history[t] = self.contamination_rate
        self.cost_history[t] = cost
        
        return self.alpha, self.contamination_rate, cost


class FarmerRiskControlModel:
    """
    Agent-Based Model for simulating farmers' risk control behaviors.
    """
    def __init__(self, num_farmers=100, time_steps=50, seed=None):
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
        """
        if seed is not None:
            np.random.seed(seed)
            
        self.num_farmers = num_farmers
        self.time_steps = time_steps
        
        # Model parameters
        self.f = [0, 0, 0, 0, 0]  # Penalties at test points and from illness
        self.beta = [0, 0, 0, 0]  # Testing probabilities
        self.P = 0.5  # Probability of identifying eligible products
        self.c_e_range = (0.1, 1.0)  # Range for effort cost
        self.c_k_range = (0.1, 1.0)  # Range for technology cost
        
        # Initialize farmers with random initial values
        self.farmers = []
        for i in range(num_farmers):
            initial_alpha = np.random.uniform(0.2, 0.8)
            initial_technology = np.random.uniform(0.2, 0.8)
            
            farmer = Farmer(i, time_steps, alpha=initial_alpha, technology_level=initial_technology)
            self.farmers.append(farmer)
            
        # Initialize arrays to store aggregate results
        self.mean_alpha_history = np.zeros(time_steps)
        self.mean_contamination_history = np.zeros(time_steps)
        self.mean_cost_history = np.zeros(time_steps)
    
    def set_parameters(self, f, beta, P):
        """
        Set model parameters.
        
        Parameters:
        -----------
        f : list of float
            Penalties at test points 1-4 and from illness (f1, f2, f3, f4, f5)
        beta : list of float
            Testing probabilities at test points 1-4 (β1, β2, β3, β4)
        P : float
            Probability that farmer's eligible products can be identified
        """
        self.f = f
        self.beta = beta
        self.P = P
    
    def run_simulation(self):
        """
        Run the simulation for the specified number of time steps.
        """
        for t in range(self.time_steps):
            alphas = []
            contamination_rates = []
            costs = []
            
            for farmer in self.farmers:
                # Random costs for effort and technology for each farmer
                c_e = np.random.uniform(self.c_e_range[0], self.c_e_range[1])
                c_k = np.random.uniform(self.c_k_range[0], self.c_k_range[1])
                
                # Get neighboring farmers (simple example: 2 random neighbors)
                neighbor_indices = np.random.choice(
                    [i for i in range(self.num_farmers) if i != farmer.id], 
                    size=min(2, self.num_farmers-1), 
                    replace=False
                )
                neighbors = [self.farmers[i] for i in neighbor_indices]
                
                # Update farmer
                alpha, contamination_rate, cost = farmer.update(t, self.f, self.beta, self.P, c_e, c_k, neighbors)
                
                alphas.append(alpha)
                contamination_rates.append(contamination_rate)
                costs.append(cost)
            
            # Store aggregate results
            self.mean_alpha_history[t] = np.mean(alphas)
            self.mean_contamination_history[t] = np.mean(contamination_rates)
            self.mean_cost_history[t] = np.mean(costs)
            
            print(f"Time step {t}: Mean α = {self.mean_alpha_history[t]:.4f}, " +
                  f"Mean contamination = {self.mean_contamination_history[t]:.4f}, " +
                  f"Mean cost = {self.mean_cost_history[t]:.4f}")
    
    def plot_results(self):
        """
        Plot the simulation results.
        """
        time_steps_range = range(self.time_steps)
        
        # Create a figure with 3 subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        
        # Plot mean alpha over time
        axs[0].plot(time_steps_range, self.mean_alpha_history)
        axs[0].set_xlabel('Time Step')
        axs[0].set_ylabel('Mean Risk Control Effort (α)')
        axs[0].set_title('Mean Risk Control Effort over Time')
        axs[0].grid(True)
        
        # Plot mean contamination rate over time
        axs[1].plot(time_steps_range, self.mean_contamination_history)
        axs[1].set_xlabel('Time Step')
        axs[1].set_ylabel('Mean Contamination Rate (σ)')
        axs[1].set_title('Mean Contamination Rate over Time')
        axs[1].grid(True)
        
        # Plot mean cost over time
        axs[2].plot(time_steps_range, self.mean_cost_history)
        axs[2].set_xlabel('Time Step')
        axs[2].set_ylabel('Mean Cost')
        axs[2].set_title('Mean Cost over Time')
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('simulation_results.png')
        plt.close()
        
        # Additional plot: distribution of final alpha values
        plt.figure(figsize=(10, 6))
        final_alphas = [farmer.alpha_history[self.time_steps-1] for farmer in self.farmers]
        plt.hist(final_alphas, bins=20)
        plt.xlabel('Final Risk Control Effort (α)')
        plt.ylabel('Number of Farmers')
        plt.title('Distribution of Final Risk Control Effort')
        plt.grid(True)
        plt.savefig('alpha_distribution.png')
        plt.close()
        
        # Plot individual farmer trajectories (sample of 10 farmers)
        plt.figure(figsize=(10, 6))
        for i in range(min(10, self.num_farmers)):
            plt.plot(time_steps_range, self.farmers[i].alpha_history, label=f'Farmer {i}')
        plt.xlabel('Time Step')
        plt.ylabel('Risk Control Effort (α)')
        plt.title('Individual Farmer Risk Control Effort Trajectories')
        plt.legend()
        plt.grid(True)
        plt.savefig('individual_trajectories.png')
        plt.close()


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
    
    # Plot the results
    model.plot_results() 
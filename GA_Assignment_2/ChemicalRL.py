import random
import numpy as np

class ChemicalRL:
    def __init__(self, num_chemicals, bounds, total_constraint, costs, learning_rate=0.1, gamma=0.9, epsilon=0.1, episodes=1000):
        # Initialize parameters
        self.num_chemicals = num_chemicals
        self.bounds = bounds
        self.total_constraint = total_constraint
        self.costs = costs
        self.learning_rate = learning_rate  # Learning rate for Q-learning
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.episodes = episodes  # Number of episodes for learning
        self.q_table = np.zeros((num_chemicals, len(bounds)))  # Q-table initialization (actions vs. states)

    def _choose_action(self, state):
        """Choose an action based on epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.num_chemicals - 1)  # Explore: choose random action
        else:
            return np.argmax(self.q_table[state])  # Exploit: choose the best action (max Q-value)

    def _get_reward(self, solution):
        """Calculate the reward (negative cost and penalty for constraint violation)."""
        cost = sum(solution[i] * self.costs[i] for i in range(self.num_chemicals))
        penalty = 10 * abs(self.total_constraint - sum(solution))  # Penalty if constraint is violated
        return -cost - penalty  # We want to minimize cost, hence negative reward

    def _update_q_table(self, state, action, reward, next_state):
        """Update the Q-table based on the Q-learning formula."""
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state, action] = (1 - self.learning_rate) * self.q_table[state, action] + \
                                      self.learning_rate * (reward + self.gamma * self.q_table[next_state, best_next_action])

    def train(self):
        """Train the RL agent over multiple episodes."""
        for episode in range(self.episodes):
            solution = [random.uniform(lb, ub) for lb, ub in self.bounds]  # Random initial solution
            self._normalize(solution)

            for t in range(1000):  # Limit the number of steps per episode
                state = tuple(solution)  # Use the entire solution as the state
                action = self._choose_action(state)

                # Apply action (modify solution)
                # Here, you would define the action based on the problem. In this example, just modify the solution randomly.
                solution[action] += random.uniform(-0.1, 0.1)  # Example action, can be modified for specific problem
                self._normalize(solution)

                reward = self._get_reward(solution)
                next_state = tuple(solution)  # Updated solution as the next state

                self._update_q_table(state, action, reward, next_state)

                if sum(solution) > self.total_constraint:  # If total constraint is violated, stop episode
                    break

    def _normalize(self, solution):
        """Ensure that the solution satisfies the total constraint."""
        total = sum(solution)
        if total != self.total_constraint:
            scale_factor = self.total_constraint / total
            for i in range(len(solution)):
                solution[i] *= scale_factor
                # Ensure the values stay within bounds
                solution[i] = max(self.bounds[i][0], min(self.bounds[i][1], solution[i]))

    def optimize(self, solution=None):
        """Optimize the solution using the trained RL agent."""
        if solution is None:
            solution = [random.uniform(lb, ub) for lb, ub in self.bounds]  # Random initial solution
        self._normalize(solution)
        
        for _ in range(1000):  # Number of optimization iterations
            state = tuple(solution)  # Use the entire solution as the state
            action = self._choose_action(state)
            
            solution[action] += random.uniform(-0.1, 0.1)  # Modify solution based on action
            self._normalize(solution)

            reward = self._get_reward(solution)
            next_state = tuple(solution)

            # Update Q-table using the reward
            self._update_q_table(state, action, reward, next_state)

        return solution, self._get_reward(solution)  # Return the optimized solution and its reward

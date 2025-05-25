import sys
import os
import numpy as np

# Add the src directory to the path
# Get the absolute path to the project root directory (RL_project_sailing)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add both the project root and src directory to Python path
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

# Import the BaseAgent class
from src.agents.base_agent import BaseAgent
from src.env_sailing import SailingEnv
from src.initial_windfields import get_initial_windfield



class QLearningAgent(BaseAgent):
    """A simple Q-learning agent for the sailing environment using only local information."""
    
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1,
                 num_angle_bins=8, velocity_bins=None, wind_bins=None, goal_dist_bins=None):
        super().__init__()
        self.np_random = np.random.default_rng()
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        # Discretization parameters
        self.num_angle_bins = num_angle_bins
        self.velocity_bins = velocity_bins if velocity_bins is not None else [0.2, 0.5, 1, 2, 5]
        self.wind_bins = wind_bins if wind_bins is not None else [0.5, 1, 2, 5]
        self.goal_dist_bins = goal_dist_bins if goal_dist_bins is not None else [5, 10, 20, 30, 45]
        
        self.num_actions = 9  # Move this line up before estimate_state_action_space
        
        # Initialize Q-table
        # State space: position_x, position_y, velocity_direction, wind_direction
        # Action space: 9 possible actions
        self.q_table = {}
        
        # Track state-action space size
        self.estimate_state_action_space()
        
    def discretize_state(self, observation):
        """Convert continuous observation to discrete state for Q-table lookup."""
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]
        goal_x, goal_y = 16, 32

        # Velocity
        v_angle = (np.arctan2(vy, vx) + 2 * np.pi) % (2 * np.pi)
        v_angle_bin = int(v_angle / (2 * np.pi) * self.num_angle_bins)
        v_norm = np.sqrt(vx**2 + vy**2)
        v_norm_bin = np.digitize([v_norm], self.velocity_bins)[0]

        # Wind
        w_angle = (np.arctan2(wy, wx) + 2 * np.pi) % (2 * np.pi)
        w_angle_bin = int(w_angle / (2 * np.pi) * self.num_angle_bins)
        w_norm = np.sqrt(wx**2 + wy**2)
        w_norm_bin = np.digitize([w_norm], self.wind_bins)[0]

        # Goal vector
        gx, gy = goal_x - x, goal_y - y
        g_angle = (np.arctan2(gy, gx) + 2 * np.pi) % (2 * np.pi)
        g_angle_bin = int(g_angle / (2 * np.pi) * self.num_angle_bins)
        g_norm = np.sqrt(gx**2 + gy**2)
        g_norm_bin = np.digitize([g_norm], self.goal_dist_bins)[0]

        return (v_angle_bin, v_norm_bin, w_angle_bin, w_norm_bin, g_angle_bin, g_norm_bin)
        
    def act(self, observation):
        """Choose an action using epsilon-greedy policy."""
        # Discretize the state
        state = self.discretize_state(observation)
        
        # Epsilon-greedy action selection
        if self.np_random.random() < self.exploration_rate:
            # Explore: choose a random action
            return self.np_random.integers(0, 9)
        else:
            # Exploit: choose the best action according to Q-table
            if state not in self.q_table:
                # If state not in Q-table, initialize it
                self.q_table[state] = np.zeros(9)
            
            # Return action with highest Q-value
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state):
        """Update Q-table based on observed transition."""
        # Initialize Q-values if states not in table
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(9)
        
        # Q-learning update
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
    
    def reset(self):
        """Reset the agent for a new episode."""
        # Nothing to reset for Q-learning agent
        pass
        
    def seed(self, seed=None):
        """Set the random seed."""
        self.np_random = np.random.default_rng(seed)
        
    def save(self, path):
        """Save the Q-table to a file."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)
            
    def load(self, path):
        """Load the Q-table from a file."""
        import pickle
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)

    def estimate_state_action_space(self):
        num_v_angle = self.num_angle_bins
        num_v_norm = len(self.velocity_bins) + 1
        num_w_angle = self.num_angle_bins
        num_w_norm = len(self.wind_bins) + 1
        num_g_angle = self.num_angle_bins
        num_g_norm = len(self.goal_dist_bins) + 1

        num_states = num_v_angle * num_v_norm * num_w_angle * num_w_norm * num_g_angle * num_g_norm
        num_state_action_pairs = num_states * self.num_actions

        print(f"Estimated number of discrete states: {num_states}")
        print(f"Estimated number of state-action pairs: {num_state_action_pairs}")
        self.num_states = num_states
        self.num_state_action_pairs = num_state_action_pairs

ql_agent_full = QLearningAgent(
    learning_rate=0.1, 
    discount_factor=0.95, 
    exploration_rate=0.3,
    num_angle_bins=12,  # Increase from 8 to 12
    velocity_bins=[0.1, 0.3, 0.6, 1.0, 1.5, 2.5, 4.0],  # More bins
    wind_bins=[0.3, 0.7, 1.2, 2.0, 3.5],  # More bins
    goal_dist_bins=[3, 7, 12, 18, 25, 35, 45]  # More bins
)

# Set fixed seed for reproducibility
np.random.seed(42)
ql_agent_full.seed(42)

# Training parameters
num_episodes = {
    'training_1': 1000,  # Back from 2000
    'training_2': 200,   # Back from 500  
    'training_3': 1000   # Back from 1500
}
max_steps = 1_000
windfields = ['training_1', 'training_3', 'training_2']  # Hard to easy

# Progress tracking (per windfield)
rewards_history = {wf: [] for wf in windfields}
steps_history = {wf: [] for wf in windfields}
success_history = {wf: [] for wf in windfields}

# Training loop
print("Starting focused training...")
print(f"Training episodes per windfield:")
for wf, eps in num_episodes.items():
    print(f"- {wf}: {eps} episodes")

import time
start_time = time.time()

for windfield in windfields:
    print(f"\nTraining on windfield: {windfield}")
    
    # Create environment with current windfield
    env = SailingEnv(**get_initial_windfield(windfield))
    
    # Adjust exploration rate for each windfield
    if windfield in ['training_1', 'training_3']:
        ql_agent_full.exploration_rate = 0.3  # Start with higher exploration for challenging windfields
    else:
        ql_agent_full.exploration_rate = 0.2  # Less exploration needed for easier windfield
    
    for episode in range(num_episodes[windfield]):
        # Reset environment and get initial state
        observation, info = env.reset(seed=episode)
        state = ql_agent_full.discretize_state(observation)
        
        total_reward = 0
        
        for step in range(max_steps):
            # Select action and take step
            action = ql_agent_full.act(observation)
            next_observation, reward, done, truncated, info = env.step(action)
            next_state = ql_agent_full.discretize_state(next_observation)
            
            # Update Q-table
            ql_agent_full.learn(state, action, reward, next_state)
            
            # Update state and total reward
            state = next_state
            observation = next_observation
            total_reward += reward
            
            # Break if episode is done
            if done or truncated:
                break
        
        # Record metrics for this windfield
        rewards_history[windfield].append(total_reward)
        steps_history[windfield].append(step+1)
        success_history[windfield].append(done)
        
        # Update exploration rate (decrease over time)
        # Slower decay for challenging windfields
        if windfield in ['training_1', 'training_3']:
            ql_agent_full.exploration_rate = max(0.1, ql_agent_full.exploration_rate * 0.995)  # Back to original
        else:
            ql_agent_full.exploration_rate = max(0.05, ql_agent_full.exploration_rate * 0.98)  # Back to original
        
        # Print progress every 20 episodes
        if (episode + 1) % 20 == 0:
            recent_success_rate = sum(success_history[windfield][-20:]) / 20 * 100
            recent_steps = np.mean(steps_history[windfield][-20:])
            print(f"Windfield {windfield}, Episode {episode+1}/{num_episodes[windfield]}: " +
                  f"Success rate (last 20): {recent_success_rate:.1f}%, " +
                  f"Avg steps (last 20): {recent_steps:.1f}")

training_time = time.time() - start_time

# Calculate overall statistics
print("\nTraining Summary:")
print(f"Total training time: {training_time:.1f} seconds")

for windfield in windfields:
    success_rate = sum(success_history[windfield]) / len(success_history[windfield]) * 100
    avg_steps = np.mean(steps_history[windfield])
    print(f"\nWindfield {windfield}:")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Min steps: {min(steps_history[windfield])}")
    print(f"Max steps: {max(steps_history[windfield])}")

print(f"\nQ-table size: {len(ql_agent_full.q_table)} states")

# Plotting training progress
import matplotlib.pyplot as plt

# Calculate rolling averages for each windfield
window_size = 20  # Increased window size for smoother curves
fig, axes = plt.subplots(2, 1, figsize=(12, 12))
fig.suptitle('Training Progress by Windfield')

for windfield in windfields:
    # Calculate moving averages
    rolling_steps = np.convolve(steps_history[windfield], 
                               np.ones(window_size)/window_size, mode='valid')
    rolling_success = np.convolve([1 if s else 0 for s in success_history[windfield]], 
                                 np.ones(window_size)/window_size, mode='valid') * 100
    
    # Plot steps
    axes[0].plot(rolling_steps, label=windfield)
    axes[0].set_ylabel('Average Steps')
    axes[0].set_title(f'Steps per Episode ({window_size}-episode rolling average)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot success rate
    axes[1].plot(rolling_success, label=windfield)
    axes[1].set_ylabel('Success Rate (%)')
    axes[1].set_xlabel('Episode')
    axes[1].set_title(f'Success Rate ({window_size}-episode rolling average)')
    axes[1].legend()
    axes[1].grid(True)

plt.tight_layout()
plt.show()

# Testing phase
print("\nTesting the trained agent on all windfields...")
ql_agent_full.exploration_rate = 0

# Test parameters
num_test_episodes = 20  # Increased number of test episodes
max_steps = 1000

for windfield in windfields:
    print(f"\nTesting on windfield: {windfield}")
    test_env = SailingEnv(**get_initial_windfield(windfield))
    
    test_steps = []
    test_success = []
    
    # Testing loop
    for episode in range(num_test_episodes):
        # Reset environment
        observation, info = test_env.reset(seed=2000 + episode)  # Different seeds from training
        
        total_reward = 0
        
        for step in range(max_steps):
            # Select action using learned policy
            action = ql_agent_full.act(observation)
            observation, reward, done, truncated, info = test_env.step(action)
            
            total_reward += reward
            
            # Break if episode is done
            if done or truncated:
                break
        
        test_steps.append(step + 1)
        test_success.append(done)
        
        print(f"Test Episode {episode+1}: Steps={step+1}, " +
              f"Position={info['position']}, Goal reached={done}")
    
    # Print summary statistics for this windfield
    success_rate = sum(test_success) / len(test_success) * 100
    avg_steps = np.mean(test_steps)
    print(f"\nTest Summary for {windfield}:")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Min steps: {min(test_steps)}")
    print(f"Max steps: {max(test_steps)}")

# Save our trained agent
from src.utils.agent_utils import save_enhanced_qlearning_agent

save_enhanced_qlearning_agent(
    agent=ql_agent_full,
    output_path="src/agents/QLearningAgent.py"
)
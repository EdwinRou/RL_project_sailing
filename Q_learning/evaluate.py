import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

# Add the src directory to the path
# Get the absolute path to the project root directory (RL_project_sailing)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add both the project root and src directory to Python path
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))


from src.test_agent_validity import validate_agent, load_agent_class
from src.evaluation import evaluate_agent, visualize_trajectory
from initial_windfields import get_initial_windfield, INITIAL_WINDFIELDS

# Path to your agent implementation (change this to your agent file path)
# AGENT_PATH = "../src/agents/agent_naive.py"
AGENT_PATH = "src/agents/QLearningAgent.py"

# Choose which training initial windfields to evaluate on
TRAINING_INITIAL_WINDFIELDS = ["training_1", "training_2", "training_3"]

# Evaluation parameters for all initial windfields
ALL_SEEDS = [42, 43, 44, 45, 46]  # Seeds to use for all evaluations
ALL_MAX_HORIZON = 200             # Maximum steps per episode



def load_and_validate_agent(agent_path):
    """Load and validate an agent from a file path."""
    try:
        # Validate the agent first
        validation_results = validate_agent(agent_path)
        
        if not validation_results['valid']:
            print("❌ Agent validation failed:")
            for error in validation_results['errors']:
                print(f"  - {error}")
            return None
        
        # If valid, return the agent class
        return validation_results['agent_class']
        
    except Exception as e:
        print(f"❌ Error loading agent: {str(e)}")
        return None


def print_evaluation_results(results):
    """Print evaluation results in a readable format."""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Steps: {results['mean_steps']:.1f} ± {results['std_steps']:.1f}")
    
    if 'individual_results' in results:
        print("\nIndividual Episode Results:")
        for i, episode in enumerate(results['individual_results']):
            print(f"  Seed {episode['seed']}: " + 
                  f"Reward={episode['reward']:.1f}, " +
                  f"Steps={episode['steps']}, " +
                  f"Success={'✓' if episode['success'] else '✗'}")
    
    print("="*50)

# Load and validate the agent specified in AGENT_PATH
AgentClass = load_and_validate_agent(AGENT_PATH)

if AgentClass:
    print(f"✅ Successfully loaded agent: {AgentClass.__name__}")
    # Create an instance of your agent
    agent = AgentClass()
else:
    print("⚠️ Please fix your agent implementation before evaluation.")



# Only run if the agent was successfully loaded
if 'agent' in locals():
    # Store results for each initial windfield
    all_results = {}
    
    print(f"Evaluating agent on {len(TRAINING_INITIAL_WINDFIELDS)} training initial windfields...")
    
    # Evaluate on each initial windfield
    for initial_windfield_name in TRAINING_INITIAL_WINDFIELDS:
        print(f"\nInitial windfield: {initial_windfield_name}")
        
        # Get the initial windfield
        initial_windfield = get_initial_windfield(initial_windfield_name)
        
        # Run the evaluation
        results = evaluate_agent(
            agent=agent,
            initial_windfield=initial_windfield,
            seeds=ALL_SEEDS,
            max_horizon=ALL_MAX_HORIZON,
            verbose=False,  # Less verbose for multiple evaluations
            render=False,
            full_trajectory=False
        )
        
        # Store results
        all_results[initial_windfield_name] = results
        
        # Print summary
        print(f"  Success Rate: {results['success_rate']:.2%}")
        print(f"  Mean Reward: {results['mean_reward']:.2f}")
        print(f"  Mean Steps: {results['mean_steps']:.1f}")
    
    # Print overall performance
    total_success = sum(r['success_rate'] for r in all_results.values()) / len(all_results)
    print("\n" + "="*50)
    print(f"OVERALL SUCCESS RATE: {total_success:.2%}")
    print("="*50)




if 'agent' in locals() and 'all_results' in locals():
    # Create summary table with pandas
    import pandas as pd
    
    # Prepare data for summary table
    summary_data = []
    for initial_windfield_name, results in all_results.items():
        summary_data.append({
            'Initial Windfield': initial_windfield_name.upper(),
            'Mean Reward': f"{results['mean_reward']:.2f} ± {results['std_reward']:.2f}",
            'Success Rate': f"{results['success_rate']:.2%}",
            'Mean Steps': f"{results['mean_steps']:.1f} ± {results['std_steps']:.1f}"
        })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Display summary table
    from IPython.display import display
    print("\nSummary of Results Across All Initial Windfields:")
    display(summary_df)
    
    # Calculate average across initial windfields
    avg_success_rate = np.mean([results['success_rate'] for results in all_results.values()])
    avg_reward = np.mean([results['mean_reward'] for results in all_results.values()])
    avg_steps = np.mean([results['mean_steps'] for results in all_results.values()])
    
    print(f"\nAverage Across Training Initial Windfields:")
    print(f"  Success Rate: {avg_success_rate:.2%}")
    print(f"  Mean Reward: {avg_reward:.2f}")
    print(f"  Mean Steps: {avg_steps:.1f}")
    print("\nNote: Your final evaluation will include hidden test initial windfields.")

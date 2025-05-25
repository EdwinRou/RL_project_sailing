import sys
import os
import numpy as np

# Add the src directory to the path
# Get the absolute path to the project root directory (RL_project_sailing)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add both the project root and src directory to Python path
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))


# Import the agent validation module
from src.test_agent_validity import validate_agent


def print_validation_results(results):
    """Print validation results in a readable format."""
    print("\n" + "="*50)
    print(f"Agent: {results['agent_name']}")
    print("="*50)
    
    if results['valid']:
        print("✅ VALID: Agent implements all required methods and returns valid actions.")
    else:
        print("❌ INVALID: Agent validation failed. See errors below.")
        
    if results['errors']:
        print("\nErrors:")
        for i, error in enumerate(results['errors'], 1):
            print(f"{i}. {error}")
    
    if results['warnings']:
        print("\nWarnings (these won't prevent submission):")
        for i, warning in enumerate(results['warnings'], 1):
            print(f"{i}. {warning}")
    
    print("="*50)

# your_agent_path = "../your_agent.py"  # Change this!
your_agent_path = "src/agents/QLearningAgent.py"

try:
    # Validate your agent
    results = validate_agent(your_agent_path)
    print_validation_results(results)
    
except FileNotFoundError:
    print(f"❌ File not found: {your_agent_path}")
    print("Please provide the correct path to your agent implementation.")
except Exception as e:
    print(f"❌ Error: {str(e)}")
    print("Make sure your file has valid Python syntax and can be imported.")
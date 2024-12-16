# Medical Diagnosis Path Planning with Inverse Reinforcement Learning

This project implements an attention-based inverse reinforcement learning approach for medical diagnosis path planning. The system learns from expert demonstrations to generate optimal diagnostic pathways while maintaining interpretability.

## Project Structure

- `inverse_rl_main.py`: Main training script for the IRL agent.
- `policy_gradient_main.py`: Main training script for the policy gradient agent.
- `plot_loss.py`: Script for plotting the training results.
- `create_toy_dataset.py`: Script for creating a synthetic toy medical dataset.
- `synthetic_medical_data.csv`: Synthetic toy medical dataset for training and testing.
- `Training.csv`: Kaggle dataset.
- `converted_training.csv`: Converted Kaggle training data for the RL agent.


## Requirements

- Python 3.8+
- PyTorch
- NetworkX
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Running the code
- Train using IRL approach:
python inverse_rl_main.py

- Train using policy gradient approach:
python policy_gradient_main.py

- Create new dataset:
python create_dataset.py

(Note: My GitHub got some connection issues, so I uploaded the code zip on the BB submission. I did not change any code after BB submission.)

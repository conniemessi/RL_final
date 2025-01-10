import torch
import torch.optim as optim
from perception_agent import PerceptionAgent
from reasoning_agent import ReasoningAgent
from environment import DiagnosisEnvironment
from training import train_two_stage_rl
from torch.optim.lr_scheduler import LambdaLR

def main():
    # ----------------------------
    # 1. Environment Setup
    # ----------------------------
    symptom_set = ['fever', 'cough', 'headache', 'fatigue', 'nausea']
    disease_set = ['Common Cold', 'Flu', 'Migraine', 'Gastroenteritis']  # Removed Unknown
    
    diagnosis_rules = {
        'Common Cold': ['cough', 'fever'],
        'Flu': ['fever', 'cough', 'fatigue'],
        'Migraine': ['headache', 'nausea'],
        'Gastroenteritis': ['nausea', 'fatigue']
    }
    
    # Initialize the environment with the toy dataset
    env = DiagnosisEnvironment(symptom_set, disease_set, diagnosis_rules, dataset_path='toy_dataset.json')
    
    # ----------------------------
    # 2. Agent Initialization
    # ----------------------------
    # Parameters
    n_symptoms = len(symptom_set)  # 5
    n_diseases = len(disease_set)  # 4
    hidden_size = 128
    temperature = 1.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize Perception Agent
    perception_agent = PerceptionAgent(
        input_size=n_symptoms,
        hidden_size=hidden_size,
        output_size=n_symptoms,
        num_heads=4,
        dropout=0.1
    )
    
    # Initialize Reasoning Agent with new parameters
    reasoning_agent = ReasoningAgent(
        num_symptoms=n_symptoms,
        num_diseases=n_diseases,
        hidden_size=hidden_size,
        temperature=temperature,
        device=device
    )
    
    # ----------------------------
    # 3. Optimizer Setup
    # ----------------------------
    learning_rate = 1e-4  # Reduced learning rate for transformer
    combined_params = list(perception_agent.parameters()) + list(reasoning_agent.parameters())
    optimizer = optim.Adam(combined_params, lr=learning_rate, betas=(0.9, 0.98))
    
    # Add warmup scheduler for transformer
    def warmup_lambda(epoch):
        warmup_steps = 100
        return min((epoch + 1) ** (-0.5), epoch * (warmup_steps ** (-1.5)))
    
    perception_scheduler = LambdaLR(optimizer, warmup_lambda)
    
    # ----------------------------
    # 4. Training Configuration
    # ----------------------------
    num_episodes = 100
    batch_size = 4  # Adjust based on dataset size
    gamma = 0.99  # Discount factor
    
    # ----------------------------
    # 5. Start Training
    # ----------------------------
    train_two_stage_rl(
        env=env,
        perception_agent=perception_agent,
        reasoning_agent=reasoning_agent,
        optimizer=optimizer,
        num_episodes=num_episodes,
        batch_size=batch_size,
        beta=0.5  # Adjust this to control KL term importance
    )
    
    # ----------------------------
    # 6. Save Trained Models
    # ----------------------------
    torch.save(perception_agent.state_dict(), 'perception_agent.pth')
    torch.save(reasoning_agent.state_dict(), 'reasoning_agent.pth')
    
    print("Training completed and models saved.")

if __name__ == "__main__":
    main()


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch.distributions import Categorical
import pandas as pd
import ast

class ExpertPolicy(nn.Module):
    """Expert (doctor) policy network with attention"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ExpertPolicy, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, available_actions):
        attention_weights = []
        policy_outputs = []
        
        for action in available_actions:
            # Compute attention weight
            weight = self.attention(action)
            attention_weights.append(weight)
            
            # Compute policy output
            policy_output = self.policy_net(action)
            policy_outputs.append(policy_output)
        
        attention_weights = torch.stack(attention_weights)
        attention_weights = torch.softmax(attention_weights, dim=0)
        
        policy_outputs = torch.stack(policy_outputs).squeeze(-1)
        
        # Combine attention weights with policy outputs
        weighted_outputs = policy_outputs * attention_weights.squeeze(-1)
        
        return weighted_outputs, attention_weights

class RewardNet(nn.Module):
    """Learned reward function"""
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(RewardNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.node_embeddings = {}

    def add_node(self, node, embedding):
        self.graph.add_node(node)
        self.node_embeddings[node] = embedding

        
    def add_edge(self, node1, node2):
        self.graph.add_edge(node1, node2)

def create_knowledge_graph_from_data(df, symptoms, diagnoses):
    embedding_dim = 64
    kg = KnowledgeGraph()
    node_mapping = {}
    
    # Add symptom nodes
    for i, symptom in enumerate(symptoms):
        embedding = torch.randn(embedding_dim)  # Random initialization
        kg.add_node(symptom, embedding)
        node_mapping[symptom] = i
    
    # Add diagnosis nodes
    for i, diagnosis in enumerate(diagnoses):
        embedding = torch.randn(embedding_dim)
        kg.add_node(diagnosis, embedding)
        node_mapping[diagnosis] = len(symptoms) + i
    
    # Add edges based on co-occurrence
    for _, row in df.iterrows():
        # Connect symptoms to each other
        for s1 in row['symptoms']:
            for s2 in row['symptoms']:
                if s1 != s2:
                    kg.add_edge(s1, s2)
            # Connect symptoms to diagnosis
            kg.add_edge(s1, row['diagnosis'])
    
    return kg, node_mapping

def collect_expert_demonstrations(df, kg, num_episodes=100):
    demonstrations = []
    for _ in range(num_episodes):
        case = df.iloc[np.random.randint(len(df))]
        
        # Create path from symptoms to diagnosis
        path = case['symptoms'].copy()
        path.append(case['diagnosis'])
        
        # Convert path to state-action pairs
        for i in range(len(path)-1):
            state = path[:i+1]
            action = path[i+1]
            demonstrations.append({
                'state': state,
                'action': action,
                'next_state': path[:i+2]
            })
    
    return demonstrations

def find_tensor_index(tensor_list, target_tensor):
    for i, tensor in enumerate(tensor_list):
        if torch.all(tensor == target_tensor):
            return i
    return -1

def train_irl(expert_demos, kg, expert_policy, reward_net, num_epochs=100):
    expert_optimizer = optim.Adam(expert_policy.parameters())
    reward_optimizer = optim.Adam(reward_net.parameters())
    
    for epoch in range(num_epochs):
        # Train reward network
        reward_loss = 0
        for demo in expert_demos:
            # Convert list of state embeddings to tensor and ensure correct dimensions
            state_embeddings = [kg.node_embeddings[s] for s in demo['state']]
            state = torch.stack(state_embeddings).mean(dim=0)
            action = kg.node_embeddings[demo['action']]
            
            # Ensure action is a tensor
            action = action.unsqueeze(0)  # Add batch dimension
            state = state.unsqueeze(0)    # Add batch dimension
            
            # Get available actions
            current_node = demo['state'][-1]
            available_actions = [kg.node_embeddings[n] for n in kg.graph.neighbors(current_node)]
            
            # Ensure available_actions are tensors
            available_actions = [a.clone() for a in available_actions]  # Create copies of tensors
            
            # Expert policy prediction
            policy_output, attention_weights = expert_policy(state, available_actions)
            
            # Compute reward for expert action
            expert_reward = reward_net(state, action)
            
            # Sample non-expert action
            non_expert_node = np.random.choice(list(kg.graph.nodes()))
            non_expert_action = kg.node_embeddings[non_expert_node].unsqueeze(0)  # Add batch dimension
            non_expert_reward = reward_net(state, non_expert_action)
            
            # Maximum entropy IRL loss
            reward_loss += torch.max(torch.zeros_like(expert_reward), 
                        non_expert_reward - expert_reward)
        
        reward_optimizer.zero_grad()
        reward_loss.backward()
        reward_optimizer.step()
        
        # Train expert policy
        policy_loss = 0
        for demo in expert_demos:
            state = torch.stack([kg.node_embeddings[s] for s in demo['state']]).mean(dim=0)
            action = kg.node_embeddings[demo['action']]
            
            current_node = demo['state'][-1]
            available_actions = [kg.node_embeddings[n] for n in kg.graph.neighbors(current_node)]
            
            policy_output, attention_weights = expert_policy(state, available_actions)
            target_idx = find_tensor_index(available_actions, action)
            policy_loss += nn.CrossEntropyLoss()(policy_output.unsqueeze(0), 
                                               torch.tensor([target_idx]))
        
        expert_optimizer.zero_grad()
        policy_loss.backward()
        expert_optimizer.step()
        
        # Consider adding regularization or normalizing the policy outputs
        # Also might want to scale down the loss:
        policy_loss = policy_loss / len(expert_demos)  # Average over demonstrations
        
        # if epoch % 10 == 0:
        print(f"Epoch {epoch}")
        print(f"Reward Loss: {reward_loss.item():.4f}")
        print(f"Policy Loss: {policy_loss.item():.4f}")

def evaluate_policy(policy, kg, test_cases):
    correct = 0
    total = 0
    
    for case in test_cases:
        try:
            # Convert state (symptoms) to tensor properly
            if not case['symptoms']:  # Check if symptoms list is empty
                continue
                
            # Create state tensor from symptoms
            state_embeddings = [kg.node_embeddings[s] for s in case['symptoms']]
            if not state_embeddings:  # Check if we got any embeddings
                continue
                
            # Stack embeddings and get mean
            state_tensor = torch.stack(state_embeddings)
            state_tensor = state_tensor.mean(dim=0)
            
            # Ensure state tensor has correct shape
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension if needed
            
            # Get available actions
            available_actions = get_available_actions(kg, state_tensor)
            
            # Get policy prediction
            with torch.no_grad():
                action_probs = policy(state_tensor, available_actions)
                predicted_action = torch.argmax(action_probs).item()
            
            # Compare with actual diagnosis
            if predicted_action == case['diagnosis']:
                correct += 1
            total += 1
            
        except Exception as e:
            print(f"Error processing case: {e}")
            continue
    
    accuracy = correct / total if total > 0 else 0
    return accuracy

# Helper function to get available actions
def get_available_actions(kg, state):
    # Get all possible next nodes from current state
    current_nodes = state.nonzero().squeeze().tolist()
    if isinstance(current_nodes, int):
        current_nodes = [current_nodes]
    
    available_actions = set()
    for node in current_nodes:
        neighbors = list(kg.graph.neighbors(node))
        available_actions.update(neighbors)
    
    # Convert to tensor
    action_embeddings = [kg.node_embeddings[a] for a in available_actions]
    if not action_embeddings:
        return torch.zeros((1, kg.node_embeddings[0].size(0)))
    return torch.stack(action_embeddings)

def visualize_knowledge_graph(kg, symptoms, diagnoses):
    plt.figure(figsize=(12, 8))
    
    # Create a layout for the graph
    pos = nx.spring_layout(kg.graph, seed=42)
    
    # Draw the full graph
    nx.draw(kg.graph, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_weight='bold')
    
    # Highlight symptom and diagnosis nodes differently
    symptom_nodes = [node for node in kg.graph.nodes if node in symptoms]
    diagnosis_nodes = [node for node in kg.graph.nodes if node in diagnoses]
    nx.draw_networkx_nodes(kg.graph, pos, nodelist=symptom_nodes, node_color='lightblue', node_size=500)
    nx.draw_networkx_nodes(kg.graph, pos, nodelist=diagnosis_nodes, node_color='lightgreen', node_size=500)
    
    plt.title("Knowledge Graph Visualization")
    plt.show()

    
def main():
    # Load or create dataset
    df = pd.read_csv('converted_training.csv')  # You'll need to provide this
    df['symptoms'] = df['symptoms'].apply(ast.literal_eval)
    
    # Split symptoms and diagnoses
    symptoms = list(set([item for sublist in df['symptoms'] for item in sublist]))
    diagnoses = list(df['diagnosis'].unique())
    
    # Create knowledge graph
    kg, node_mapping = create_knowledge_graph_from_data(df, symptoms, diagnoses)
    
    # Visualize the initial knowledge graph structure
    visualize_knowledge_graph(kg, symptoms, diagnoses)

    # Initialize networks
    state_dim = 64  # Same as embedding dimension
    action_dim = 64
    hidden_dim = 128
    output_dim = len(diagnoses)
    
    expert_policy = ExpertPolicy(state_dim, hidden_dim, output_dim)
    reward_net = RewardNet(state_dim, action_dim, hidden_dim)
    
    # Collect expert demonstrations
    demonstrations = collect_expert_demonstrations(df, kg)
    
    # Train using IRL
    train_irl(demonstrations, kg, expert_policy, reward_net)
    
    # Evaluate
    test_df = df.sample(frac=0.2)  # 20% for testing
    # accuracy = evaluate_policy(expert_policy, kg, test_df.to_dict('records'))
    # print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from validation import MedicalGuidelineValidator, ValidationMetrics
from create_dataset import create_toy_dataset

class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_embeddings = {}
        
    def add_node(self, node_id, features):
        self.graph.add_node(node_id)
        self.node_embeddings[node_id] = features
        
    def add_edge(self, source, target, relation):
        self.graph.add_edge(source, target, relation=relation)

class AttentionPathFinder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=4)
        
        # Policy network
        self.policy_network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU()
        )
        
        # Action probability layer
        self.action_head = nn.Linear(hidden_dim, embedding_dim)
        
    def forward(self, state, available_actions):
        # Ensure state has correct embedding dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension
            if state.size(-1) != self.embedding_dim:
                # Pad or project state to match embedding dimension
                state = F.pad(state, (0, self.embedding_dim - state.size(-1)))
        
        # Convert state and available_actions to the same dtype
        state = state.to(torch.float32)
        available_actions = available_actions.to(torch.float32)
        
        # Reshape inputs to match expected dimensions (seq_len, batch, embed_dim)
        query = state.view(1, 1, self.embedding_dim)  # [1, 1, embedding_dim]
        keys = available_actions.view(-1, 1, self.embedding_dim)  # [num_actions, 1, embedding_dim]
        values = available_actions.view(-1, 1, self.embedding_dim)  # [num_actions, 1, embedding_dim]
        
        attended_state, attention_weights = self.attention(query, keys, values)
        
        # Get policy logits
        hidden = self.policy_network(attended_state.squeeze(0))
        action_logits = self.action_head(hidden)
        
        # Mask unavailable actions
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs, attention_weights

class DiagnosisAgent:
    def __init__(self, knowledge_graph, embedding_dim, hidden_dim, learning_rate=0.001):
        self.knowledge_graph = knowledge_graph
        self.policy = AttentionPathFinder(embedding_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
    def get_action(self, state, available_actions):
        action_probs, attention_weights = self.policy(state, available_actions)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action, action_dist.log_prob(action), attention_weights
    
    def update_policy(self, rewards, log_probs):
        policy_loss = []
        for r, lp in zip(rewards, log_probs):
            policy_loss.append(-lp * r)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
    def compute_reward(self, path, target_diagnosis, human_path):
        reward = 0
        
        # Larger diagnosis reward
        if path[-1] == target_diagnosis:
            reward += 20  # Increase from 10
        
        # Intermediate rewards for correct steps
        for step in path:
            if step in human_path:
                reward += 1  # Add step-wise rewards
        
        # Smaller length penalty
        reward -= 0.05 * len(path)  # Decrease from 0.1
        
        return reward

def train_episode(agent, initial_symptoms, target_diagnosis, human_path, max_path_length):
    state = torch.tensor(initial_symptoms, dtype=torch.float)
    path = []
    log_probs = []
    rewards = []
    
    while len(path) < max_path_length:
        available_actions = get_available_actions(agent.knowledge_graph, state)
        action, log_prob, attention = agent.get_action(state, available_actions)
        
        path.append(action.item())
        log_probs.append(log_prob)
        
        if is_terminal_state(action, target_diagnosis):
            break
            
        state = get_next_state(action)
    
    reward = agent.compute_reward(path, target_diagnosis, human_path)
    rewards = [reward] * len(log_probs)  # Distribute reward across all actions
    
    agent.update_policy(rewards, log_probs)
    return path, reward

# Create knowledge graph from dataset
def create_knowledge_graph_from_data(df, symptoms, diagnoses):
    kg = MedicalKnowledgeGraph()
    
    # Add symptom nodes
    embedding_dim = 64
    for symptom in symptoms:
        # Create random embedding for symptom
        embedding = torch.randn(embedding_dim)
        kg.add_node(symptom, embedding)
    
    # Add diagnosis nodes
    for diagnosis in diagnoses:
        # Create random embedding for diagnosis
        embedding = torch.randn(embedding_dim)
        kg.add_node(diagnosis, embedding)
    
    # Add edges based on co-occurrence in the dataset
    for _, row in df.iterrows():
        current_symptoms = row['symptoms']
        current_diagnosis = row['diagnosis']
        
        # Add edges between symptoms and diagnosis
        for symptom in current_symptoms:
            kg.add_edge(symptom, current_diagnosis, relation='indicates')
        
        # Add edges between related symptoms
        for i in range(len(current_symptoms)):
            for j in range(i + 1, len(current_symptoms)):
                kg.add_edge(current_symptoms[i], current_symptoms[j], relation='co-occurs')
                kg.add_edge(current_symptoms[j], current_symptoms[i], relation='co-occurs')
    
    # Visualize the knowledge graph
    visualize_knowledge_graph(kg, symptoms, diagnoses)
    
    return kg

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

# Function to visualize the knowledge graph and the learned path
def visualize_path(kg, path, symptoms, diagnoses):
    plt.figure(figsize=(12, 8))
    
    # Create a layout for the graph
    pos = nx.spring_layout(kg.graph, seed=42)
    
    # Draw the full graph in light gray
    nx.draw(kg.graph, pos, with_labels=True, node_size=500, 
            node_color='lightgray', font_size=10, 
            font_weight='bold', alpha=0.3)
    
    # Highlight the path
    path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
    nx.draw_networkx_edges(kg.graph, pos, edgelist=path_edges, 
                          edge_color='red', width=2)
    
    # Add direction arrows to the path
    edge_labels = {(path[i], path[i+1]): f'{i+1}' 
                  for i in range(len(path)-1)}
    nx.draw_networkx_edge_labels(kg.graph, pos, 
                                edge_labels=edge_labels)
    
    # Highlight nodes in the path with different colors for start/end
    nx.draw_networkx_nodes(kg.graph, pos, 
                          nodelist=[path[0]], 
                          node_color='green', 
                          node_size=700, 
                          label='Start')
    nx.draw_networkx_nodes(kg.graph, pos, 
                          nodelist=[path[-1]], 
                          node_color='red', 
                          node_size=700, 
                          label='End')
    nx.draw_networkx_nodes(kg.graph, pos, 
                          nodelist=path[1:-1], 
                          node_color='orange', 
                          node_size=700, 
                          label='Path')
    
    # Add legend
    plt.legend()
    
    plt.title(f"Diagnostic Path: {' -> '.join(map(str, path))}")
    plt.axis('off')
    plt.show()

def validate_agent(agent, validator, test_cases, n_episodes=100):
    validation_metrics = ValidationMetrics(validator)
    
    for episode in range(n_episodes):
        # Sample a test case
        case = test_cases.iloc[np.random.randint(len(test_cases))]
        
        # Run episode
        path, actions = agent.run_episode(case['symptoms'])
        
        # Evaluate the episode
        metrics = validation_metrics.evaluate_path(
            path=path,
            true_diagnosis=case['diagnosis'],
            patient_symptoms=set(case['symptoms']),
            taken_actions=actions
        )
        
        if episode % 10 == 0:
            print(f"Validation Episode {episode}")
            print("Metrics:", metrics)
    
    # Get summary statistics
    summary_stats = validation_metrics.get_summary_statistics()
    return summary_stats

def get_available_actions(knowledge_graph, state):
    # Assuming state is a node in the graph, return its neighbors as available actions
    current_node = state.argmax().item()  # Get the index of the current node
    node_name = list(knowledge_graph.node_embeddings.keys())[current_node]
    neighbors = list(knowledge_graph.graph.neighbors(node_name))
    
    # Convert neighbor names to indices
    neighbor_indices = [list(knowledge_graph.node_embeddings.keys()).index(n) for n in neighbors]
    
    # Create a tensor of available actions
    available_actions = torch.stack([knowledge_graph.node_embeddings[n] for n in neighbors])
    
    # Ensure available_actions has the correct embedding dimension
    if available_actions.size(-1) != knowledge_graph.node_embeddings[node_name].size(-1):
        # Adjust the embedding dimension if necessary
        available_actions = available_actions.view(-1, knowledge_graph.node_embeddings[node_name].size(-1))
    
    return available_actions

def is_terminal_state(action, target_diagnosis):
    # Check if the action corresponds to the target diagnosis
    return action.item() == target_diagnosis

def get_next_state(action):
    # In this simplified example, the next state is the action taken
    return action

# Example usage and training
def main():
    # Load or create dataset
    df, symptoms, diagnoses = create_toy_dataset()
    
    # Create knowledge graph
    kg = create_knowledge_graph_from_data(df, symptoms, diagnoses)
    print("Knowledge graph created with {} nodes and {} edges".format(
        len(kg.graph.nodes), len(kg.graph.edges)))
    
    # Initialize agent
    embedding_dim = 64
    hidden_dim = 128
    agent = DiagnosisAgent(kg, embedding_dim, hidden_dim)
    
    # Training loop
    n_episodes = 1000
    max_path_length = 5
    
    # Split data into train and test
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    
    # Create validator
    validator = MedicalGuidelineValidator()
    
    for episode in range(n_episodes):
        # Sample a random case from dataset
        case = df.iloc[np.random.randint(len(df))]
        
        # Create initial state from symptoms
        initial_symptoms = torch.zeros(len(symptoms))
        for symptom in case['symptoms']:
            symptom_idx = symptoms.index(symptom)
            initial_symptoms[symptom_idx] = 1
        
        # Create simplified human path (just for demonstration)
        human_path = case['symptoms'] + [case['diagnosis']]
        
        # Train episode
        path, reward = train_episode(
            agent=agent,
            initial_symptoms=initial_symptoms,
            target_diagnosis=case['diagnosis'],
            human_path=human_path,
            max_path_length=max_path_length
        )
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {reward:.2f}")
            print(f"Path: {path}")
            
            # Uncomment this line to visualize
            # visualize_path(kg, path, symptoms, diagnoses)
        
        # Validate every 200 episodes
        # if episode % 200 == 0:
        #     print("\nRunning validation...")
        #     validation_stats = validate_agent(agent, validator, test_df)
        #     print("\nValidation Statistics:")
        #     for metric, stats in validation_stats.items():
        #         print(f"{metric}:")
        #         for stat_name, value in stats.items():
        #             print(f"  {stat_name}: {value:.3f}")

if __name__ == "__main__":
    main()

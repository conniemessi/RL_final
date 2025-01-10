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
import seaborn as sns
import ast


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
        
        # Create distribution only over available actions
        action_dist = torch.distributions.Categorical(action_probs[:len(available_actions)])
        
        # Sample action index within the range of available actions
        action_idx = action_dist.sample()
        
        # Ensure action index is within bounds
        action = action_idx % len(available_actions)  # This ensures we get a valid index
        
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
            reward += 20  # Keep the existing reward for correct diagnosis
        
        # Perfect path reward
        if path == human_path:
            reward += 50  # Large bonus for exactly matching the human path
        else:
            # Intermediate rewards for correct steps (existing logic)
            for step in path:
                if step in human_path:
                    reward += 10
                    
            # Additional reward for correct sequence
            for i in range(min(len(path), len(human_path))):
                if i < len(path) and path[i] == human_path[i]:
                    reward += 20  # Extra reward for steps in correct order
        
        # Smaller length penalty (keep existing)
        reward -= 0.05 * len(path)
        
        return reward


def visualize_attention_scores(attention_weights, available_actions, step):
    """
    Visualize attention scores for each decision step
    """
    plt.figure(figsize=(10, 4))
    
    # Ensure attention weights are in the correct shape
    attention = attention_weights.detach().squeeze().numpy()
    if attention.ndim == 0:  # If 0-d array
        attention = np.array([attention])  # Convert to 1-d array
    
    # Create attention score visualization
    ax = sns.barplot(
        x=list(range(len(available_actions))),
        y=attention[:len(available_actions)],  # Ensure lengths match
        palette='YlOrRd'
    )
    
    # Add value labels on top of each bar
    for i, v in enumerate(attention[:len(available_actions)]):
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    plt.title(f'Attention Scores at Step {step}')
    plt.xlabel('Available Actions')
    plt.ylabel('Attention Weight')
    plt.tight_layout()
    plt.show()


def train_episode(agent, initial_symptoms, target_diagnosis, human_path, max_path_length):
    state = torch.tensor(initial_symptoms, dtype=torch.float)
    path = []
    log_probs = []
    attention_scores = []
    
    for step in range(max_path_length):
        # Get available actions and their IDs
        available_actions, available_node_ids = get_available_actions(agent.knowledge_graph, state)
        action, log_prob, attention_weights = agent.get_action(state, available_actions)
        
        # Convert action index to actual node ID
        selected_node_id = available_node_ids[action.item()]
        
        attention_scores.append({
            'step': step,
            'weights': attention_weights,
            'available_actions': available_node_ids,  # Store actual node IDs
            'selected_action': selected_node_id,
            'target_action': human_path[step] if step < len(human_path) else None
        })
        
        path.append(selected_node_id)
        log_probs.append(log_prob)
        
        if is_terminal_state(action, target_diagnosis):
            break
            
        state = get_next_state(action)
    
    # Calculate similarity with human path
    path_similarity = len(set(path).intersection(set(human_path))) / len(human_path)
    
    reward = agent.compute_reward(path, target_diagnosis, human_path)
    rewards = [reward] * len(log_probs)
    
    agent.update_policy(rewards, log_probs)
    return path, reward, path_similarity, attention_scores

# Create knowledge graph from dataset
def create_knowledge_graph_from_data(df, symptoms, diagnoses):
    kg = MedicalKnowledgeGraph()
    
    # Add nodes with sequential IDs
    for i, symptom in enumerate(symptoms):
        embedding = torch.randn(64)  # embedding_dim = 64
        kg.add_node(i, embedding)  # Use index as node ID
    
    # Add diagnosis nodes after symptoms
    start_idx = len(symptoms)
    for i, diagnosis in enumerate(diagnoses):
        embedding = torch.randn(64)
        kg.add_node(start_idx + i, embedding)
    
    # Create node ID mapping
    node_mapping = {s: i for i, s in enumerate(symptoms)}
    node_mapping.update({d: i + len(symptoms) for i, d in enumerate(diagnoses)})
    
    # Add edges using node IDs
    for _, row in df.iterrows():
        current_symptoms = [node_mapping[s] for s in row['symptoms']]
        current_diagnosis = node_mapping[row['diagnosis']]
        
        for symptom_id in current_symptoms:
            kg.add_edge(symptom_id, current_diagnosis, relation='indicates')
    
    return kg, node_mapping

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
    # Add debugging information
    print("Graph nodes:", list(kg.graph.nodes()))
    print("Attempting to visualize path:", path)
    
    # Ensure all path nodes exist in graph
    valid_path = [node for node in path if node in kg.graph.nodes()]
    if len(valid_path) != len(path):
        print("Warning: Some nodes in path don't exist in graph!")
        print("Invalid nodes:", set(path) - set(kg.graph.nodes()))
        path = valid_path
    
    if len(path) < 2:
        print("Error: Not enough valid nodes to create a path")
        return
        
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
    # Get list of all nodes
    all_nodes = list(knowledge_graph.graph.nodes())
    
    # Get current node index
    current_node = state.argmax().item()
    if current_node >= len(all_nodes):
        current_node = current_node % len(all_nodes)
    
    # Get valid neighbors
    current_node_name = all_nodes[current_node]
    neighbors = list(knowledge_graph.graph.neighbors(current_node_name))
    
    # Convert to tensor and store node IDs
    available_actions = []
    available_node_ids = []  # Store the actual node IDs
    for n in neighbors:
        available_actions.append(knowledge_graph.node_embeddings[n])
        available_node_ids.append(n)
    
    return torch.stack(available_actions), available_node_ids

def is_terminal_state(action, target_diagnosis):
    # Check if the action corresponds to the target diagnosis
    return action.item() == target_diagnosis

def get_next_state(action):
    # In this simplified example, the next state is the action taken
    return action

def visualize_paths_comparison(kg, agent_path, human_path, symptoms, diagnoses):
    plt.figure(figsize=(15, 8))
    pos = nx.spring_layout(kg.graph, seed=42)
    
    # Draw the full graph in light gray
    nx.draw(kg.graph, pos, with_labels=True, node_size=500, 
            node_color='lightgray', font_size=10, 
            font_weight='bold', alpha=0.3)
    
    # Draw human path in green
    human_edges = [(human_path[i], human_path[i+1]) 
                   for i in range(len(human_path)-1)]
    nx.draw_networkx_edges(kg.graph, pos, edgelist=human_edges, 
                          edge_color='green', width=2, 
                          label='Human Path')
    
    # Draw agent path in red
    agent_edges = [(agent_path[i], agent_path[i+1]) 
                   for i in range(len(agent_path)-1)]
    nx.draw_networkx_edges(kg.graph, pos, edgelist=agent_edges, 
                          edge_color='red', width=2, 
                          label='Agent Path', style='dashed')
    
    # Highlight nodes
    nx.draw_networkx_nodes(kg.graph, pos, 
                          nodelist=[human_path[0]], 
                          node_color='lightgreen', 
                          node_size=700, 
                          label='Start')
    nx.draw_networkx_nodes(kg.graph, pos, 
                          nodelist=[human_path[-1]], 
                          node_color='darkgreen', 
                          node_size=700, 
                          label='Target Diagnosis')
    
    # Add legend
    plt.legend()
    
    plt.title("Path Comparison: Human Expert vs Agent")
    plt.text(0.05, -0.1, 
            f"Human Path: {' -> '.join(map(str, human_path))}\n"
            f"Agent Path: {' -> '.join(map(str, agent_path))}", 
            transform=plt.gca().transAxes)
    plt.axis('off')
    plt.show()


def analyze_attention_pattern(attention_scores, human_path):
    """
    Analyze attention patterns and their alignment with human decisions
    """
    total_steps = len(attention_scores)
    correct_attention = 0
    
    print("\nDetailed Attention Analysis:")
    for attn in attention_scores:
        step = attn['step']
        # Convert 0-d array to 1-d array if necessary
        weights = attn['weights'].detach().squeeze().numpy()
        if weights.ndim == 0:  # If 0-d array
            weights = np.array([weights])  # Convert to 1-d array
            
        selected = attn['selected_action']
        target = attn['target_action']
        
        print(f"\nStep {step}:")
        print(f"Attention distribution:")
        for i, w in enumerate(weights):
            print(f"  Node {i}: {w:.3f}")
        
        if target is not None:
            print(f"Selected node: {selected}")
            print(f"Target node: {target}")
            if selected == target:
                correct_attention += 1
                print("✓ Correct attention")
            else:
                print("✗ Incorrect attention")
    
    accuracy = correct_attention / total_steps
    print(f"\nOverall attention accuracy: {accuracy:.2f}")
    
    return accuracy

# Example usage and training
def main():
    # Load or create dataset
    # df, symptoms, diagnoses = create_toy_dataset()
    
    df = pd.read_csv('synthetic_medical_data.csv')
    # Ensure symptoms are properly parsed from string to list
    df['symptoms'] = df['symptoms'].apply(ast.literal_eval)
    symptoms = list(set([item for sublist in df['symptoms'] for item in sublist]))
    diagnoses = list(df['diagnosis'].unique())
    print(f"Loaded dataset with {len(df)} cases")
    print(f"Found {len(symptoms)} unique symptoms and {len(diagnoses)} unique diagnoses")

    # Create knowledge graph with proper node IDs
    kg, node_mapping = create_knowledge_graph_from_data(df, symptoms, diagnoses)
    print(f"Knowledge graph created with {len(kg.graph.nodes)} nodes and {len(kg.graph.edges)} edges")
    print("Node mapping:", node_mapping)
    
    # Visualize the initial knowledge graph structure
    visualize_knowledge_graph(kg, symptoms, diagnoses)
    
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
        
        # Create human path (expert demonstration)
        # human_path = [symptoms.index(s) for s in case['symptoms']]
        # human_path.append(len(symptoms) + diagnoses.index(case['diagnosis']))
        
        human_path = []
        
        # Start with the most significant/severe symptoms first
        sorted_symptoms = sorted(case['symptoms'], 
                               key=lambda x: len(kg.graph[symptoms.index(x)]),  # Sort by connection count
                               reverse=True)  # Most connected symptoms first
        
        # Add symptom nodes to path
        for symptom in sorted_symptoms:
            symptom_idx = symptoms.index(symptom)
            human_path.append(symptom_idx)
            
        # Add the final diagnosis node
        diagnosis_idx = len(symptoms) + diagnoses.index(case['diagnosis'])
        human_path.append(diagnosis_idx)

        # Train episode
        path, reward, similarity, attention_scores = train_episode(
            agent=agent,
            initial_symptoms=initial_symptoms,
            target_diagnosis=case['diagnosis'],
            human_path=human_path,
            max_path_length=max_path_length
        )
        
        if episode % 100 == 0:
            print(f"\nEpisode {episode}")
            print(f"Reward: {reward:.2f}")
            print(f"Path Similarity: {similarity:.2f}")
            print("Human Path:", human_path)
            print("Agent Path:", path)
            visualize_paths_comparison(kg, path, human_path, symptoms, diagnoses)

            for attn in attention_scores:
                print(f"\nStep {attn['step']}:")
                print(f"Selected node: {attn['selected_action']}")
                print(f"Target node: {attn['target_action']}")
                visualize_attention_scores(
                    attn['weights'],
                    attn['available_actions'],
                    attn['step']
                )

            accuracy = analyze_attention_pattern(attention_scores, human_path)
        # Validate every 200 episodes
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

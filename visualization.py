import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import deque
import torch
import networkx as nx

class VisualizationManager:
    def __init__(self, symptom_set, disease_set, window_size=100):
        self.symptom_set = symptom_set
        self.disease_set = disease_set
        self.window_size = window_size
        
        # Create index to disease name mapping
        self.idx_to_disease = {i: disease for i, disease in enumerate(disease_set)}
        
        # Initialize tracking variables
        self.rewards_history = deque(maxlen=window_size)
        self.perception_losses = deque(maxlen=window_size)
        self.reasoning_losses = deque(maxlen=window_size)
        self.episodes = []
        
        # Setup plots
        plt.style.use('seaborn-v0_8-darkgrid')
        self.setup_plots()
        
        # Initialize the graph for path visualization
        self.setup_diagnosis_graph()
    
    def setup_plots(self):
        """Initialize the plotting framework"""
        self.fig = plt.figure(figsize=(15, 8))
        
        # Create grid of subplots
        # Top row: attention matrix and path visualization
        self.ax_attention = plt.subplot2grid((2, 3), (0, 0), colspan=1)
        self.ax_path = plt.subplot2grid((2, 3), (0, 1), colspan=2)
        
        # Bottom row: loss and reward plots
        self.ax_loss = plt.subplot2grid((2, 3), (1, 0), colspan=2)
        self.ax_reward = plt.subplot2grid((2, 3), (1, 2))
        
        # Initialize attention plot
        self.ax_attention.set_title('Attention Weights', pad=10)
        
        # Initialize path plot
        self.ax_path.set_title('Diagnosis Path')
        self.ax_path.axis('off')
        
        # Initialize loss plot
        self.ax_loss.set_title('Training Losses')
        self.ax_loss.set_xlabel('Episode')
        self.ax_loss.set_ylabel('Loss')
        
        # Initialize reward plot
        self.ax_reward.set_title('Average Reward')
        self.ax_reward.set_xlabel('Episode')
        self.ax_reward.set_ylabel('Reward')
    
    def setup_diagnosis_graph(self):
        """Initialize the diagnosis graph structure"""
        self.G = nx.DiGraph()
        
        # Add symptom nodes
        for symptom in self.symptom_set:
            self.G.add_node(symptom, layer=0)
        
        # Add disease nodes
        for disease in self.disease_set:
            self.G.add_node(disease, layer=1)
        
        # Store node positions for consistent layout
        self.pos = {}
        
        # Position symptoms on the left
        for i, symptom in enumerate(self.symptom_set):
            self.pos[symptom] = (-1, (len(self.symptom_set)-1)/2 - i)
        
        # Position diseases on the right
        for i, disease in enumerate(self.disease_set):
            self.pos[disease] = (1, (len(self.disease_set)-1)/2 - i)
    
    def update_attention_matrix(self, attention_mask):
        """Update the attention matrix visualization"""
        self.ax_attention.clear()
        self.ax_attention.set_title('Attention Weights', pad=10)
        
        # Reshape attention mask for visualization
        attention_mask = attention_mask.detach().cpu().numpy()
        attention_mask = attention_mask.reshape(1, -1)
        
        # Create clean heatmap without bars
        sns.heatmap(attention_mask,
                   ax=self.ax_attention,
                   cmap='YlOrRd',
                   xticklabels=self.symptom_set,
                   yticklabels=[''],
                   cbar=False,
                   annot=True,
                   fmt='.3f',
                   annot_kws={'size': 8})
        
        # Rotate x-axis labels for better readability
        plt.setp(self.ax_attention.get_xticklabels(), rotation=45, ha='right')
    
    def update_diagnosis_path(self, attention_mask, predicted_disease_idx, true_disease_idx, reward, current_symptoms):
        """Update the diagnosis path visualization"""
        self.ax_path.clear()
        self.ax_path.set_title('Diagnosis Path')
        
        # Convert disease indices to names
        predicted_disease = self.idx_to_disease[predicted_disease_idx]
        true_disease = self.idx_to_disease[true_disease_idx]
        
        # Reset edges
        self.G.remove_edges_from(list(self.G.edges()))
        
        # Get attention weights
        attention_weights = attention_mask.detach().cpu().numpy()
        # Add edges based on attention weights
        for i, (symptom, weight) in enumerate(zip(self.symptom_set, attention_weights[0])):
            # print(f"Symptom: {symptom}, Weight: {weight}")
            if weight > 0.3:  # Only show strong connections
                self.G.add_edge(symptom, predicted_disease, weight=float(weight))
        
        # Draw the graph
        # Draw all nodes in light blue first
        nx.draw_networkx_nodes(self.G, self.pos, 
                             node_color='lightblue',
                             node_size=1500,
                             ax=self.ax_path)
        
        # Edges with varying width based on attention weights
        edges = self.G.edges(data=True)
        if edges:
            weights = [d['weight'] for (u, v, d) in edges]
            nx.draw_networkx_edges(self.G, self.pos, 
                                 edge_color='gray',
                                 width=[w * 3 for w in weights],
                                 ax=self.ax_path)
        
        # Add symptom labels on the left with present symptoms highlighted
        for i, (symptom, pos) in enumerate(self.pos.items()):
            if symptom in self.symptom_set:  # Only for symptoms
                is_present = current_symptoms[i].item() == 1
                color = 'red' if is_present else 'gray'
                self.ax_path.text(pos[0] - 0.2, pos[1], 
                                f"{symptom} ({'+' if is_present else '-'})", 
                                fontsize=8, ha='right', va='center',
                                color=color)
        
        # Add disease labels on the right
        for disease, pos in self.pos.items():
            if disease in self.disease_set:  # Only for diseases
                self.ax_path.text(pos[0] + 0.2, pos[1], disease, 
                                fontsize=8, ha='left', va='center')
        
        # Draw nodes without default labels
        nx.draw_networkx_nodes(self.G, self.pos, 
                             node_color='lightblue',
                             node_size=1500,
                             ax=self.ax_path)
        
        # Highlight predicted disease
        color = 'green' if predicted_disease == true_disease else 'red'
        nx.draw_networkx_nodes(self.G, self.pos,
                             nodelist=[predicted_disease],
                             node_color=color,
                             node_size=1500,
                             ax=self.ax_path)
        
        # Add legend showing current case details
        present_symptoms = [self.symptom_set[i] for i, val in enumerate(current_symptoms) if val.item() == 1]
        legend_text = (
            f'Current Symptoms: {", ".join(present_symptoms)}\n'
            f'Predicted: {predicted_disease}\n'
            f'Reward: {reward:.2f}'
        )
        if predicted_disease != true_disease:
            legend_text += f'\nTrue: {true_disease}'
        
        bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9)
        self.ax_path.text(0.5, -1.5, legend_text,
                         horizontalalignment='center',
                         bbox=bbox_props,
                         transform=self.ax_path.transData)
    
    def update_metrics(self, episode, reward, perception_loss=None, reasoning_loss=None):
        """Update training metrics"""
        self.episodes.append(episode)
        self.rewards_history.append(reward)
        
        if perception_loss is not None:
            self.perception_losses.append(perception_loss)
        if reasoning_loss is not None:
            self.reasoning_losses.append(reasoning_loss)
        
        # Update loss plot
        self.ax_loss.clear()
        self.ax_loss.set_title('Training Losses')
        if len(self.perception_losses) > 0:
            self.ax_loss.plot(self.episodes[-len(self.perception_losses):], 
                            self.perception_losses, label='Perception Loss (KL)')
        if len(self.reasoning_losses) > 0:
            self.ax_loss.plot(self.episodes[-len(self.reasoning_losses):], 
                            self.reasoning_losses, label='Reasoning Loss (Recon)')
        self.ax_loss.legend()
        self.ax_loss.set_xlabel('Episode')
        self.ax_loss.set_ylabel('Loss')
        
        # Update reward plot
        self.ax_reward.clear()
        self.ax_reward.set_title('Average Reward')
        self.ax_reward.plot(self.episodes[-len(self.rewards_history):], 
                          self.rewards_history, 'g-')
        self.ax_reward.set_xlabel('Episode')
        self.ax_reward.set_ylabel('Reward')
    
    def update_visualization(self, episode, attention_mask, predicted_disease, true_disease, 
                           reward, current_symptoms, perception_loss=None, reasoning_loss=None):
        """Update all visualizations"""
        self.update_attention_matrix(attention_mask)
        self.update_diagnosis_path(attention_mask, predicted_disease, true_disease, reward, current_symptoms)
        self.update_metrics(episode, reward, perception_loss, reasoning_loss)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
    
    def save_visualization(self, filename='diagnosis_visualization.png'):
        """Save current visualization to file"""
        plt.savefig(filename, bbox_inches='tight', dpi=300) 
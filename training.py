import random
import torch
import torch.nn.functional as F
from visualization import VisualizationManager
import numpy as np

def train_two_stage_rl(env, perception_agent, reasoning_agent, optimizer, num_episodes=100, vis_manager=None):
    """
    Train both perception and reasoning agents using RL
    
    Args:
        env: The diagnosis environment
        perception_agent: The perception agent model
        reasoning_agent: The reasoning agent model
        optimizer: The optimizer for training both agents
        num_episodes: Number of training episodes
        vis_manager: Optional visualization manager
    """
    experience_buffer = ExperienceBuffer()
    vis_manager = VisualizationManager(env.symptom_set, env.disease_set, env.diagnosis_rules)
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        episode_losses = []
        cases_solved = 0
        total_cases = env.num_cases
        # while not done:
        while cases_solved < total_cases and env.current_index < env.num_cases:
            # Forward pass through perception agent
            current_symptoms = state['current_symptoms'].view(-1, 1, len(env.symptom_set))
            symptom_history = state['symptom_history'].view(-1, 1, len(env.symptom_set))
            state_perception = torch.cat([current_symptoms, symptom_history], dim=1)
            attention_dist = perception_agent.predict_mask(state_perception)
            # Apply attention to symptoms
            masked_symptoms = state['current_symptoms'] * attention_dist.squeeze()
            
            # Forward pass through rule-based reasoning agent
            disease_probs, rule_weights = reasoning_agent(masked_symptoms.unsqueeze(0))  # Add batch dimension
            predicted_disease = torch.argmax(disease_probs).item()
            # Update environment's symptom history with current attention
            env.symptom_history = np.maximum(
                env.symptom_history, 
                attention_dist.detach().cpu().numpy()[0]
            )

            # Environment step
            next_state, reward, done, info = env.step(disease_probs)
            
            # Create prior attention from diagnosis rules
            prior_attention = create_prior_attention(state, info['true_disease'], env)
            # Compute VAE loss
            recon_loss = F.cross_entropy(
                disease_probs.view(1, -1), 
                torch.tensor([info['true_disease']], dtype=torch.long)
            )
            
            kl_loss = torch.sum(
                attention_dist * torch.log(attention_dist / prior_attention + 1e-10)
            )
            
            # Total ELBO loss
            beta = 0.5
            total_loss = recon_loss + beta * kl_loss
            
            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            episode_losses.append({
                'total_loss': total_loss.item(),
                'recon_loss': recon_loss.item(),
                'kl_loss': kl_loss.item()
            })
            
            # Store experience and update state
            experience_buffer.add((
                state_perception,
                attention_dist,
                reward,
                next_state,
                done,
                info['true_disease']
            ))
            
            state = next_state
            total_reward += reward
            if reward > 0:
                cases_solved += 1
        
            # Episode summary
            avg_losses = {k: np.mean([loss[k] for loss in episode_losses]) 
                        for k in episode_losses[0].keys()}
            
            # Break if all cases processed
            if env.current_index >= env.num_cases:
                break

            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"Total Reward: {total_reward}")
            print(f"Average Losses:")
            print(f"Total Loss: {avg_losses['total_loss']:.4f}")
            print(f"Reconstruction Loss: {avg_losses['recon_loss']:.4f}")
            print(f"KL Loss: {avg_losses['kl_loss']:.4f}\n")

            
            # Update visualization with actual loss values
            vis_manager.update_visualization(
                episode=episode,
                attention_mask=attention_dist,
                predicted_disease=predicted_disease,
                true_disease=info['true_disease'],
                reward=total_reward,
                current_symptoms=state['current_symptoms'],
                perception_loss=avg_losses['kl_loss'],
                reasoning_loss=avg_losses['recon_loss'],
                rule_weights=rule_weights
            )
        vis_manager.update_metrics(episode, total_reward, avg_losses['kl_loss'], avg_losses['recon_loss'])
    # Save final visualization
    vis_manager.save_visualization('final_diagnosis_visualization.png')

def create_prior_attention(state, true_disease_idx, env):
    """Create prior attention distribution based on diagnosis rules"""
    true_disease = env.disease_set[true_disease_idx]
    relevant_symptoms = env.diagnosis_rules[true_disease]
    
    prior = torch.zeros_like(state['current_symptoms'])
    for i, symptom in enumerate(env.symptom_set):
        if symptom in relevant_symptoms:
            prior[i] = 1.0
    
    # Add small epsilon to avoid division by zero
    prior = prior + 1e-10
    # Normalize to create probability distribution
    prior = prior / prior.sum()
    return prior

class ExperienceBuffer:
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size
    
    def add(self, experience):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer) 
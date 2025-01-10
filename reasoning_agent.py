import torch
import torch.nn as nn
import torch.nn.functional as F

class ReasoningAgent(nn.Module):
    def __init__(self, num_symptoms, num_diseases, hidden_size, temperature=1.0, device='cuda'):
        super(ReasoningAgent, self).__init__()
        self.num_symptoms = num_symptoms
        self.num_diseases = num_diseases
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.device = device

        # Predicate embeddings (for symptoms)
        self.predicate_embedding = nn.Embedding(num_symptoms + 1, hidden_size)
        self.predicate_embedding.weight.data.copy_(torch.eye(hidden_size)[:num_symptoms + 1])
        
        # Rule embeddings (for each disease's rules)
        self.rule_embedding = nn.Embedding(num_diseases * num_symptoms, hidden_size)
        
        # Initialize embeddings
        self.init_embeddings()
        
        # Move to device
        self.to(device)

    def init_embeddings(self):
        # Initialize rule embeddings with small random values
        nn.init.xavier_uniform_(self.rule_embedding.weight)

    def compute_rule_matching(self, masked_symptoms):
        """
        Compute similarity scores between rules and predicates
        Args:
            masked_symptoms: tensor of shape [batch_size, num_symptoms]
        """
        # Get predicate embeddings for all symptoms
        predicate_indices = torch.arange(self.num_symptoms).to(self.device)
        predicate_emb = self.predicate_embedding(predicate_indices)  # [num_symptoms, hidden_size]
        
        # Get rule embeddings for all diseases
        rule_indices = torch.arange(self.num_diseases * self.num_symptoms).to(self.device)
        rule_emb = self.rule_embedding(rule_indices)  # [num_diseases * num_symptoms, hidden_size]
        
        # Compute similarity scores
        # W = softmax(Q_f K^T / τ)
        similarity_scores = torch.mm(rule_emb, predicate_emb.T) / self.temperature  # [num_diseases * num_symptoms, num_symptoms]
        W = F.softmax(similarity_scores, dim=1)  # [num_diseases * num_symptoms, num_symptoms]
        
        return W

    def forward(self, masked_symptoms):
        """
        Predict disease based on masked symptoms using rule matching
        Args:
            masked_symptoms: tensor of shape [batch_size, num_symptoms]
        Returns:
            disease_probs: tensor of shape [batch_size, num_diseases]
        """
        batch_size = masked_symptoms.shape[0]
        
        # Get matching weights
        W = self.compute_rule_matching(masked_symptoms)  # [num_diseases * num_symptoms, num_symptoms]
        
        # Reshape W to separate disease rules
        W = W.view(self.num_diseases, self.num_symptoms, -1)  # [num_diseases, num_symptoms, num_symptoms]
        
        # Compute φ_f(O_masked) = ∏_{j∈O_masked} w_{ij}v_j
        # v_j is 1 if symptom j is in O_masked, 0 otherwise
        v = masked_symptoms.unsqueeze(1)  # [batch_size, 1, num_symptoms]
        
        # Compute rule satisfaction scores
        rule_scores = torch.prod(torch.where(v > 0, W, torch.ones_like(W)), dim=-1)  # [num_diseases, num_symptoms]
        
        # Sum over rules for each disease
        disease_scores = torch.sum(rule_scores, dim=1)  # [num_diseases]
        
        # Convert to probabilities
        disease_probs = F.softmax(disease_scores, dim=0)
        print(f"Disease scores: {disease_scores}")
        
        # Expand for batch size
        disease_probs = disease_probs.unsqueeze(0).expand(batch_size, -1)
        
        return disease_probs
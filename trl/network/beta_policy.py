import torch
from torch import nn
from transformers import AutoModelForCausalLM
from torch.nn import functional as F

class BetaPolicyModel(nn.Module):
    def __init__(self, base_model_name, trust_remote_code, beta_hidden_dim=2048):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=trust_remote_code)
        self.beta_head = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, beta_hidden_dim),
            nn.ReLU(),
            nn.Linear(beta_hidden_dim, 2)  # Two outputs: alpha and beta for beta distribution
        )

    def forward(self, input_ids, attention_mask, position_ids, return_dict, output_hidden_states):
        outputs = self.base_model( # Get the base model's outputs. odict_keys(['logits', 'past_key_values', 'hidden_states'])
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            return_dict=return_dict,
            output_hidden_states=output_hidden_states,  # Ensure hidden states are returned
        )
        
        hidden_states = outputs.hidden_states[-1]  # Access the last hidden state. torch.Size([1, 117, 2048]). [batch_size, sequence_length, hidden_size]
        last_token_hidden = hidden_states[:, -1, :]  # Use the last token's hidden state for sequence representation. torch.Size([1, 2048]). [batch_size, hidden_size]

        # Compute alpha and beta parameters using the beta head
        beta_params = self.beta_head(last_token_hidden) # torch.Size([1, 2])
        
        alpha = F.softplus(beta_params[:, 0]) + 1e-8 # torch.Size([1])
        beta = F.softplus(beta_params[:, 1]) + 1e-8 # torch.Size([1])
        
        
        return alpha, beta

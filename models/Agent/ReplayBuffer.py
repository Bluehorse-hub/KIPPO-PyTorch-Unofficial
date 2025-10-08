import torch

class ReplayBuffer:
    def __init__(self, capacity, latent_dim, action_dim, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.current_states = torch.zeros((capacity, latent_dim), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, latent_dim), dtype=torch.float32, device=device)
        self.current_actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.values = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

    def add(self, current_state, current_action, reward, log_prob, value, done):
        idx = self.ptr

        self.current_states[idx] = current_state
        self.current_actions[idx] = current_action
        self.rewards[idx] = reward
        self.log_probs[idx] = log_prob
        self.values[idx] = value
        self.dones[idx] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get(self):
        return self.current_states[:self.size], self.current_actions[:self.size], self.rewards[:self.size], self.log_probs[:self.size], self.values[:self.size], self.dones[:self.size]
    
    def clear(self):
        self.ptr = 0
        self.size = 0
    
    def __len__(self):
        return self.size
import torch

class KoopmanBuffer:
    def __init__(self, capacity, state_dim, action_dim, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.current_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.current_actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)

    def add(self, current_state, next_state, current_action):
        idx = self.ptr

        self.current_states[idx] = current_state
        self.next_states[idx] = next_state
        self.current_actions[idx] = current_action

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def get(self):
        return self.current_states[:self.size], self.next_states[:self.size], self.current_actions[:self.size]
    
    def sample(self, batch_size):
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)

        batch_current_states = self.current_states[idxs]
        batch_next_states = self.next_states[idxs]
        batch_actions = self.current_actions[idxs]
        
        return batch_current_states, batch_next_states, batch_actions
    
    def clear(self):
        self.ptr = 0
        self.size = 0
    
    def __len__(self):
        return self.size

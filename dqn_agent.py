import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, input_dim, output_dim, lr=1e-3, gamma=0.99):
        self.model = DQN(input_dim, output_dim)
        self.target_model = DQN(input_dim, output_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.target_model.to(self.device)

    def act(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, 3)  # 4 discrete actions: fold, check, call, raise
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            q_values = self.model(state)
            return q_values.argmax().item()

    def train_step(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_vals = self.model(states)
        next_q_vals = self.target_model(next_states).detach()

        q_val = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)
        max_next_q_val = next_q_vals.max(1)[0]
        target = rewards + self.gamma * max_next_q_val * (1 - dones)

        loss = self.loss_fn(q_val, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.update_target()

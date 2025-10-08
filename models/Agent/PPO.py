import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.Agent.RewardNormalizer import RewardNormalizer
from debug.debug import Debug

class PPO(object):
    def __init__(self, args, env, actor, critic, buffer):

        self.args = args

        self.env = env

        self.actor = actor

        self.critic = critic
        
        self.buffer = buffer

        self.last_value = 0.0

        self.episode_reward = 0.0

        self.reward_log = []
        self.actor_loss_log = []
        self.critic_loss_log = []
        self.entropy_log = []

        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()),lr=self.args.agent_lr)

        self.debug = Debug()
        self.reward_normalizer = RewardNormalizer()

    def compute_gae(self, rewards, values, dones):
        T = len(rewards)
        advantages = torch.zeros(T, dtype=torch.float32, device=rewards.device)
        gae = 0
        values = values.squeeze(-1)

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = self.last_value * (1.0 - dones[t])
            else:
                next_value = values[t + 1] * (1.0 - dones[t])
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.args.gamma * next_value * mask - values[t]
            gae = delta + self.args.gamma * self.args.lam * gae * mask
            advantages[t] = gae
            
        returns = advantages + values

        return advantages, returns
    
    def save_model(self, path):
        actor_path = os.path.join(path, "actor.pth")
        critic_path = os.path.join(path, "critic.pth")

        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def train(self):

        states, actions, rewards, log_probs, values, dones = self.buffer.get()

        advantages, returns = self.compute_gae(rewards, values, dones)

        returns = returns.detach()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset = TensorDataset(states, actions, log_probs, returns, advantages, values)
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)

        num_batchs = len(loader)

        for epoch in range(self.args.epochs):

            epoch_actor_loss = 0.0
            epoch_critic_loss = 0.0
            epoch_entropy = 0.0

            for batch_states, batch_actions, batch_log_prob, batch_returns, batch_advantages, batch_values in loader:

                #*--- V Training ---*#
                current_v = self.critic(batch_states).squeeze(-1)

                # Value Clipping
                batch_values = batch_values.squeeze(-1)
                current_v_clipped = batch_values + torch.clamp(current_v - batch_values, -self.args.epsilon, self.args.epsilon)

                original_loss = F.mse_loss(current_v, batch_returns, reduction="none")
                clipped_loss = F.mse_loss(current_v_clipped, batch_returns, reduction="none")

                critic_loss = torch.max(original_loss, clipped_loss).mean()

                #*--- Policy Training ---*#
                policy = self.actor.policy(batch_states)
                new_log_probs = policy.log_prob(batch_actions).sum(dim=-1)

                entropy = policy.entropy().mean()

                batch_log_prob = batch_log_prob.squeeze(-1)
                ratio = torch.exp(new_log_probs - batch_log_prob)

                surrogate_objective1 = batch_advantages * ratio
                surrogate_objective2 = torch.clamp(ratio, 1 - self.args.epsilon, 1 + self.args.epsilon) * batch_advantages
                actor_loss = -(torch.min(surrogate_objective1, surrogate_objective2).mean() + self.args.entropy_coef * entropy)

                total_loss = actor_loss + self.args.value_coef * critic_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), max_norm=0.5
                )
                self.optimizer.step()

                epoch_actor_loss += actor_loss.item()
                epoch_critic_loss += critic_loss.item()
                epoch_entropy += entropy.item()

            self.actor_loss_log.append(epoch_actor_loss / num_batchs)
            self.critic_loss_log.append(epoch_critic_loss / num_batchs)
            self.entropy_log.append(epoch_entropy / num_batchs)

        return self.actor_loss_log, self.critic_loss_log, self.entropy_log
    
    def rollout(self, rollout_step=2048):
        state, _ = self.env.reset()
        state = torch.from_numpy(state).to(self.args.cuda, torch.float32)

        for step in range(rollout_step):

            # Observation Clipping
            state = torch.clamp(state, -10.0, 10.0)
            
            with torch.no_grad():
                action, log_prob, action_tanh = self.actor.sample_action(state)
                value = self.critic(state)

            next_state, reward, terminated, truncated, _ = self.env.step(action_tanh.cpu().numpy())
            next_state = torch.from_numpy(next_state).to(self.args.cuda, torch.float32)

            done = terminated or truncated
            reward = float(np.squeeze(reward))
            self.episode_reward += reward

            # reward normalization
            self.reward_normalizer.update(reward)
            reward = self.reward_normalizer.normalize(reward)

            self.buffer.add(state, next_state, action, reward, log_prob, value, done)

            state = next_state

            if done:
                self.reward_log.append(self.episode_reward)
                self.episode_reward = 0.0
                state, _ = self.env.reset()
                state = torch.from_numpy(state).to(self.args.cuda, torch.float32)

        with torch.no_grad():
            self.last_value = self.critic(state)


    def linear_lr_scheduler(self, current_update, total_updates):
        frac = 1.0 - (current_update / total_updates)
        lr_now = self.args.agent_lr * frac
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr_now
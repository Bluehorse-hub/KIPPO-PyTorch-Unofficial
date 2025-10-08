import gymnasium as gym
import faulthandler
import argparse
import torch
import numpy as np
from tqdm import tqdm

from utils.manager import ExperimentManager 
from debug.debug import Debug
from models.Koopman.StateAutoEncoder import StateEncoder, StateDecoder
from models.Koopman.ActionEncoder import ActionEncoder
from models.Koopman.ControlMatrix import ControlMatrix
from models.Koopman.StateTransitionMatrix import StateTransitionMatrix
from models.Koopman.KoopmanBuffer import KoopmanBuffer
from models.Koopman.Koopman import Koopman

from models.Agent.PPO import PPO
from models.Agent import Actor
from models.Agent import Critic
from models.Agent.ReplayBuffer import ReplayBuffer

def readParser():
    parser = argparse.ArgumentParser(description='KIPPO')

    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 0)')
    
    parser.add_argument('--latent_dim', type=int, default=64, metavar='N',
                        help='latent dim (default: 64)')

    parser.add_argument('--hidden_dim', type=int, default=64, metavar='N',
                        help='hidden dim (default: 64)')
    
    parser.add_argument('--agent_lr', type=float, default=3e-4, metavar='N',
                        help='leaning rate of agent (default: 3e-4)')
    
    parser.add_argument('--gamma', type=float, default=0.99, metavar='N',
                        help='discount rate (default: 0.99)')
    
    parser.add_argument('--lam', type=float, default=0.95, metavar='N',
                        help='bias-variance tradeoff for advantage estimation (default: 0.95)')
    
    parser.add_argument('--epsilon', type=float, default=0.2, metavar='N',
                        help='Clipping epsilon for PPO surrogate objective (default: 0.2)')
    
    parser.add_argument('--entropy_coef', type=float, default=0.0, metavar='N',
                        help='Entropy regularization coefficient (default: 0.0)')
    
    parser.add_argument('--value_coef', type=float, default=0.5, metavar='N',
                        help='Value loss regularization coefficient (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size (default: 64)')
    
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='num epochs (default: 10)')
    
    parser.add_argument('--rollout_step', type=int, default=2048, metavar='N',
                        help='rollout step (default: 2048)')
    
    parser.add_argument('--num_updates', type=int, default=500, metavar='N',
                        help='num updates (default: 500)')

    parser.add_argument('--cuda', default='cuda:0',
                        help='run on CUDA (default: cuda:0)')

    return parser.parse_args()

def main(args=None):
    if args is None:
        args = readParser()

    faulthandler.enable()
    tqdm.monitor_interval = 0

    device = torch.device(args.cuda)
    latent_dim = args.latent_dim

    debug = Debug()
    manager = ExperimentManager(args)
    
    env = gym.make("HalfCheetah-v5", render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"{state_dim=}")
    print(f"{action_dim=}")

    for exp_id in range(manager.exp_num):

        debug.value("exp_id", exp_id)
        phi_x = StateEncoder(state_dim, latent_dim).to(device)
        phi_x_inv = StateDecoder(state_dim, latent_dim).to(device)
        phi_u = ActionEncoder(action_dim, latent_dim).to(device)
        B = ControlMatrix(latent_dim).to(device)
        K = StateTransitionMatrix(latent_dim).to(device)
        koopman_buffer = KoopmanBuffer(100000, state_dim, action_dim, device)

        actor = Actor.MLPPolicy(latent_dim, action_dim).to(device)
        critic = Critic.MLPCritic(latent_dim).to(device)
        rl_buffer = ReplayBuffer(10000, latent_dim, action_dim, device)

        koopman = Koopman(args, phi_x, phi_x_inv, phi_u, B, K, koopman_buffer)
        agent = PPO(args, env, actor, critic, rl_buffer)

        for update in tqdm(range(args.num_updates)):
            state, _ = env.reset()
            state = torch.from_numpy(state).to(device, torch.float32)

            for step in range(args.rollout_step):

                # Observation Clipping
                state = torch.clamp(state, -10.0, 10.0)

                with torch.no_grad():
                    latent_state = phi_x(state)
                    action, log_prob, action_tanh = actor.sample_action(latent_state)
                    value = critic(latent_state)

                next_state, reward, terminated, truncated, _ = env.step(action_tanh.cpu().numpy())
                next_state = torch.from_numpy(next_state).to(device, torch.float32)

                done = terminated or truncated
                reward = float(np.squeeze(reward))
                agent.episode_reward += reward

                # reward normalization
                agent.reward_normalizer.update(reward)
                reward = agent.reward_normalizer.normalize(reward)

                rl_buffer.add(latent_state, action, reward, log_prob, value, done)
                koopman_buffer.add(state, next_state, action)

                state = next_state

                if done:
                    agent.reward_log.append(agent.episode_reward)
                    agent.episode_reward = 0.0
                    state, _ = env.reset()
                    state = torch.from_numpy(state).to(device, torch.float32)

            with torch.no_grad():
                latent_state = phi_x(state)
                agent.last_value = critic(latent_state)
            
            koopman.train()
            agent.train()
            agent.linear_lr_scheduler(update, args.num_updates)

            manager.plot(agent.actor_loss_log, "epoch", "loss", "Actor Loss transition", "actor_loss_transition.pdf")
            manager.plot(agent.critic_loss_log, "epoch", "loss", "Critic Loss transition", "critic_loss_transition.pdf")
            manager.plot(agent.entropy_log, "epoch", "entropy", "Entropy transition", "entropy_transition.pdf")
            manager.plot(agent.reward_log, "episodes", "reward", "reward transition", "reward_transition.pdf")

            rl_buffer.clear()
            koopman_buffer.clear()

        manager.all_rewards_list.append(agent.reward_log)
        agent.save_model(manager.current_save_agent_path)
        koopman.save_model(manager.current_save_koopman_path)
        manager.next()

    manager.plot_all_rewards_transition()

if __name__ == "__main__":
    main()
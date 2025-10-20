import argparse
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch

from models.Koopman.StateAutoEncoder import StateEncoder
from models.Agent import Actor

def readParser():
    parser = argparse.ArgumentParser(description='KIPPO')

    parser.add_argument('--env', default='HalfCheetah-v5',
                        help='env name (default: HalfCheetah-v5)')
    
    parser.add_argument('--latent_dim', type=int, default=64, metavar='N',
                        help='latent dim (default: 64)')
    
    parser.add_argument('--hidden_dim', type=int, default=64, metavar='N',
                        help='hidden dim (default: 64)')
    
    parser.add_argument('--actor_weight', default='sample/actor.pth',
                        help='run on CUDA (default: cuda:0)')

    parser.add_argument('--state_encoder_weight', default='sample/state_encoder.pth',
                        help='run on CUDA (default: cuda:0)')
    
    parser.add_argument('--video_path', default='video/',
                        help='run on CUDA (default: cuda:0)')

    parser.add_argument('--cuda', default='cuda:0',
                        help='run on CUDA (default: cuda:0)')

    return parser.parse_args()

def main(args=None):
    if args is None:
        args = readParser()
    
    device = args.cuda

    env = gym.make(args.env, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=args.video_path, episode_trigger=lambda e: True)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    phi_x = StateEncoder(state_dim, args.latent_dim).to(device)
    actor = Actor.MLPPolicy(args.latent_dim, action_dim).to(device)

    phi_x.load_state_dict(torch.load(args.state_encoder_weight, weights_only=True))
    actor.load_state_dict(torch.load(args.actor_weight, weights_only=True))

    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device)

    done = False
    total_reward = 0

    while not done:
        with torch.no_grad():
            latent_state = phi_x(state)
            _, _, action_tanh = actor.sample_action(latent_state, test=True)
        state, reward, terminated, truncated, _ = env.step(action_tanh.cpu().numpy())
        state = torch.tensor(state, dtype=torch.float32, device=device)
        done = terminated or truncated
        total_reward += reward

    env.close()
    print(f"Total reward: {total_reward}")

if __name__ == "__main__":
    main()
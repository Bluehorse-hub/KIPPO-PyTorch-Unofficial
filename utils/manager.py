import os
import yaml
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt

class ExperimentManager(object):
    def __init__(self, args, exp_num=5, exp_path="experiments"):
        self.args = args
        self.time_path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_id = 0
        self.exp_num = exp_num

        self.exp_path = os.path.join(exp_path, self.time_path)
        self.current_exp_path = os.path.join(self.exp_path, str(self.experiment_id))
        self.current_save_agent_path = os.path.join(self.current_exp_path, "weight/agent")
        self.current_save_koopman_path = os.path.join(self.current_exp_path, "weight/koopman")
        self.current_save_result_path = os.path.join(self.current_exp_path, "result")

        os.makedirs(self.current_exp_path, exist_ok=True)
        os.makedirs(self.current_save_agent_path, exist_ok=True)
        os.makedirs(self.current_save_koopman_path, exist_ok=True)
        os.makedirs(self.current_save_result_path, exist_ok=True)

        hparams = vars(self.args)
        with open(os.path.join(self.exp_path, "params.yaml"), "w") as f:
            yaml.safe_dump(hparams, f, sort_keys=False, allow_unicode=True)

        self.all_rewards_list = []

    def plot(self, log: list, xlabel, ylabel, title, figname="sample.pdf"):
        fig_path = os.path.join(self.current_save_result_path, figname)
        plt.figure()
        plt.plot(log)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.savefig(fig_path)
        plt.close()

    def plot_all_rewards_transition(self):
        all_rewards = np.array(self.all_rewards_list)
        experiment_num, episode_len = all_rewards.shape

        episodes = np.arange(episode_len)

        mean_rewards = all_rewards.mean(axis=0)
        sem_rewards = all_rewards.std(axis=0) / np.sqrt(experiment_num)

        fig_path = os.path.join(self.exp_path, "all_result.pdf")

        plt.plot(episodes, mean_rewards)
        plt.fill_between(episodes, mean_rewards-sem_rewards, mean_rewards+sem_rewards, alpha=0.2)
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title("Reward Transition (Mean ± SEM)")
        plt.savefig(fig_path)
        plt.close()
    
    def next(self):
        if self.experiment_id == self.exp_num - 1:
            return
        self.experiment_id += 1

        self.current_exp_path = os.path.join(self.exp_path, str(self.experiment_id))
        self.current_save_agent_path = os.path.join(self.current_exp_path, "weight/agent")
        self.current_save_koopman_path = os.path.join(self.current_exp_path, "weight/koopman")
        self.current_save_result_path = os.path.join(self.current_exp_path, "result")

        os.makedirs(self.current_exp_path, exist_ok=True)
        os.makedirs(self.current_save_agent_path, exist_ok=True)
        os.makedirs(self.current_save_koopman_path, exist_ok=True)
        os.makedirs(self.current_save_result_path, exist_ok=True)

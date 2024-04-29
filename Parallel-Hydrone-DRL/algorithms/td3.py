import torch
from torch.distributions.normal import Normal
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models import Critic
import queue
import time
import os

class LearnerTD3(object):
    def __init__(self, config, policy_net, target_policy_net, learner_w_queue, log_dir=''):
        self.config = config
        self.update_iteration = config['update_agent_ep']
        self.batch_size = config['batch_size']
        self.gamma = config['discount_rate']
        self.tau = config['tau']
        self.device = config['device']
        self.save_dir = log_dir
        self.learner_w_queue = learner_w_queue
        self.prioritized_replay = config['replay_memory_prioritized']
        self.priority_epsilon = config['priority_epsilon']
        self.model = config['model']
        self.env_stage = config['env_stage']
        self.num_train_steps = config['num_steps_train']  # number of episodes from all agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_action = config['max_action']
        self.min_action = 0.0
        self.actor = policy_net
        self.actor_target = target_policy_net
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['actor_learning_rate'])
        self.total_it = 0

        self.critic1 = Critic(config['state_dim'], config['action_dim'], config['dense_size']).to(self.device)
        self.critic1_target = Critic(config['state_dim'], config['action_dim'], config['dense_size']).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(),  lr=config['critic_learning_rate'])

        self.critic2 = Critic(config['state_dim'], config['action_dim'], config['dense_size']).to(self.device)
        self.critic2_target = Critic(config['state_dim'], config['action_dim'], config['dense_size']).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(),  lr=config['critic_learning_rate'])


        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2




    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def _update_step(self, batch, replay_priority_queue, update_step, logs):
        self.total_it += 1
        update_time = time.time()
        # Sample replay buffer
        obs, actions, rewards, next_obs, terminals, gamma, weights, inds = batch

        obs = np.asarray(obs)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        next_obs = np.asarray(next_obs)
        terminals = np.asarray(terminals)
        weights = np.asarray(weights)
        inds = np.asarray(inds).flatten()

        obs = torch.from_numpy(obs).float().to(self.device)
        next_obs = torch.from_numpy(next_obs).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        terminals = torch.from_numpy(terminals).float().to(self.device)

        rewards = rewards.unsqueeze(1)
        terminals = terminals.unsqueeze(1)

        # Select next action according to target policy
        next_action = self.actor_target.get_action(next_obs)

        # Add Gaussian noise to target action (for exploration)
        noise = Normal(torch.zeros(next_action.shape).to(self.device), self.policy_noise).sample()
        noise = torch.clamp(noise,-self.noise_clip, self.noise_clip)
        next_action = torch.clamp((next_action + noise), self.min_action, self.max_action)

        # Compute target Q-values
        target_Q1 = self.critic1_target(next_obs, next_action)
        target_Q2 = self.critic2_target(next_obs, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = rewards + ((1.0 - terminals) * self.gamma * target_Q).detach()

        # Compute current Q-values
        current_Q1 = self.critic1(obs, actions)
        current_Q2 = self.critic2(obs, actions)

        # Compute critic loss
        critic1_loss = F.mse_loss(current_Q1, target_Q)
        critic2_loss = F.mse_loss(current_Q2, target_Q)
        critic_loss = critic1_loss + critic2_loss

        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic1(obs, self.actor.get_action(obs)).mean()

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            logs[3] = actor_loss
            # Update target networks
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
                
                
                # Send updated learner to the queue
        if update_step.value % 100 == 0:
            try:
                params = [p.data.cpu().detach().numpy() for p in self.actor.parameters()]
                self.learner_w_queue.put_nowait(params)
            except:
                pass

        # Logging
        with logs.get_lock():
            logs[4] = critic_loss
            logs[5] = time.time() - update_time


    def run(self, training_on, batch_queue, replay_priority_queue, update_step, global_episode, logs):
        torch.set_num_threads(4)
        while global_episode.value <= self.config['num_agents'] * self.config['num_episodes']:
            try:
                batch = batch_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.01)
                continue

            self._update_step(batch, replay_priority_queue, update_step, logs)
            with update_step.get_lock():
                update_step.value += 1

            if update_step.value % 10000 == 0:
                print("Training step ", update_step.value)

        with training_on.get_lock():
            training_on.value = 0

        empty_torch_queue(self.learner_w_queue)
        empty_torch_queue(replay_priority_queue)
        print("Exit learner.")

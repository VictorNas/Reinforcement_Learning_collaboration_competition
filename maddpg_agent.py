import numpy as np
import random
import copy
from collections import namedtuple, deque

#from model_test import Actor, Critic
from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 3e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.0      # L2 weight decay
UPDATE_STEPS = 10       # update every steps
NUMBER_UPDATES = 8     # number of times we update at each update stepss


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed , num_agents):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            num_agents (int): number of agents
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents = num_agents
        
        # Actors Networks (w/ Target Networks)
        self.actor_local1 = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target1 = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer1 = optim.Adam(self.actor_local1.parameters(), lr=LR_ACTOR)

        self.actor_local2 = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target2= Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer2= optim.Adam(self.actor_local2.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local1 = Critic(state_size * num_agents, action_size * num_agents, random_seed).to(device)
        self.critic_target1 = Critic(state_size * num_agents, action_size * num_agents, random_seed).to(device)
        self.critic_optimizer1 = optim.Adam(self.critic_local1.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.critic_local2 = Critic(state_size * num_agents, action_size * num_agents, random_seed).to(device)
        self.critic_target2 = Critic(state_size * num_agents, action_size * num_agents, random_seed).to(device)
        self.critic_optimizer2 = optim.Adam(self.critic_local2.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)


        # Noise process
        self.noise = OUNoise((num_agents,action_size), random_seed)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        self.tstep = 0
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state[0], action[0], reward[0], next_state[0], done[0],
                   state[1], action[1], reward[1], next_state[1], done[1])
        self.tstep += 1
        
        if self.tstep % UPDATE_STEPS == 0:
            self.tstep = 0
            for i in range(NUMBER_UPDATES):
                # Learn, if enough samples are available in memory
                if len(self.memory) > BATCH_SIZE:
                        experiences = self.memory.sample()
                        self.learn(experiences, GAMMA)
             
    def act(self, state, add_noise = True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        state_1, state_2 = state[0], state[1]
        
        self.actor_local1.eval()
        with torch.no_grad():
            action_1 = self.actor_local1(torch.unsqueeze(state_1, 0)).cpu().data.numpy()
        self.actor_local1.train()
        
        self.actor_local2.eval()
        with torch.no_grad():
            action_2 = self.actor_local2(torch.unsqueeze(state_2, 0)).cpu().data.numpy()
        self.actor_local2.train()
        
        actions = np.concatenate((action_1, action_2), 0)
        
        if add_noise:
            #action = np.random.randn(*action.shape)
            actions += self.noise.sample()
            
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states_1, actions_1, rewards_1, next_states_1, dones_1 ,\
        states_2, actions_2, rewards_2, next_states_2, dones_2  = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models

        actions_next_1 = self.actor_target1(next_states_1)
        actions_next_2 = self.actor_target2(next_states_2)
        
        # update critic 1
        Q_targets_next_1 = self.critic_target1(torch.cat((next_states_1, next_states_2), 1), 
                                             torch.cat((actions_next_1.detach(), actions_next_2.detach()), 1))
        Q_targets_1 = rewards_1 + (GAMMA * Q_targets_next_1 * (1 - dones_1))
        
        Q_expected_1 = self.critic_local1(torch.cat((states_1, states_2), 1), 
                                    torch.cat((actions_1, actions_2), 1))
        critic_loss_1 = F.mse_loss(Q_expected_1, Q_targets_1)
        self.critic_optimizer1.zero_grad()
        critic_loss_1.backward()
        self.critic_optimizer1.step()
        
        # update critic 2
        
        Q_targets_next_2 = self.critic_target2(torch.cat((next_states_1, next_states_2), 1), 
                                             torch.cat((actions_next_1.detach(), actions_next_2.detach()), 1))

        Q_targets_2 = rewards_2 + (GAMMA * Q_targets_next_2 * (1 - dones_2))

        Q_expected_2 = self.critic_local2(torch.cat((states_1, states_2), 1), 
                                    torch.cat((actions_1, actions_2), 1))
        critic_loss_2 = F.mse_loss(Q_expected_2, Q_targets_2)


        self.critic_optimizer2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer2.step()


        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred_1 = self.actor_local1(states_1)
        actions_pred_2 = self.actor_local2(states_2)
           
        # update actor 1
        actor_loss_1 = -self.critic_local1(torch.cat((states_1, states_2), 1), 
                                    torch.cat((actions_pred_1, actions_pred_2.detach()), 1)).mean()
        self.actor_optimizer1.zero_grad()
        actor_loss_1.backward()
        self.actor_optimizer1.step()
        
        # update actor 2
        actor_loss_2 = -self.critic_local2(torch.cat((states_1, states_2), 1), 
                                    torch.cat((actions_pred_1.detach(), actions_pred_2), 1)).mean()
        self.actor_optimizer2.zero_grad()
        actor_loss_2.backward()
        self.actor_optimizer2.step()
        
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local1, self.critic_target1, TAU)
        self.soft_update(self.actor_local1, self.actor_target1, TAU)
        self.soft_update(self.critic_local2, self.critic_target2, TAU)
        self.soft_update(self.actor_local2, self.actor_target2, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.1, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.seed2 = np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(*x.shape)
        #dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(len(x))
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", 
                                     field_names=["state_agent1", "action_agent1", 
                                                  "reward_agent1", "next_state_agent1", "done_agent1",
                                                 "state_agent2", "action_agent2", 
                                                  "reward_agent2", "next_state_agent2", "done_agent2"])
        self.seed = random.seed(seed)
    
    def add(self, state_agent1, action_agent1, reward_agent1, next_state_agent1, done_agent1,
           state_agent2, action_agent2, reward_agent2, next_state_agent2, done_agent2):
        """Add a new experience to memory."""
        e = self.experience(state_agent1, action_agent1, reward_agent1, next_state_agent1, done_agent1,
                           state_agent2, action_agent2, reward_agent2, next_state_agent2, done_agent2)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states_agent1 = torch.from_numpy(np.vstack([e.state_agent1 for e in experiences if e is not None])).float().to(device)
        actions_agent1 = torch.from_numpy(np.vstack([e.action_agent1 for e in experiences if e is not None])).float().to(device)
        rewards_agent1 = torch.from_numpy(np.vstack([e.reward_agent1 for e in experiences if e is not None])).float().to(device)
        next_states_agent1 = torch.from_numpy(np.vstack([e.next_state_agent1 for e in experiences if e is not None])).float().to(device)
        dones_agent1 = torch.from_numpy(np.vstack([e.done_agent1 for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        states_agent2 = torch.from_numpy(np.vstack([e.state_agent2 for e in experiences if e is not None])).float().to(device)
        actions_agent2 = torch.from_numpy(np.vstack([e.action_agent2 for e in experiences if e is not None])).float().to(device)
        rewards_agent2 = torch.from_numpy(np.vstack([e.reward_agent2 for e in experiences if e is not None])).float().to(device)
        next_states_agent2 = torch.from_numpy(np.vstack([e.next_state_agent2 for e in experiences if e is not None])).float().to(device)
        dones_agent2 = torch.from_numpy(np.vstack([e.done_agent2 for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states_agent1, actions_agent1, rewards_agent1, next_states_agent1, dones_agent1,
               states_agent2, actions_agent2, rewards_agent2, next_states_agent2, dones_agent2)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
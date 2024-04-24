import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, config):
        if config is None:
            sys.exit("Error: No configuration provided for QNetwork")
        
        if len(config['LAYERS']) < 1:
            sys.exit("Error: No hidden layers provided for QNetwork")

        super(QNetwork, self).__init__()

        layers = []

        layers.append(nn.Linear(config['STATE_SIZE'], config['LAYERS'][0]))
        for i in range(1,len(config['LAYERS'])):
            layers.append(nn.Linear(config['LAYERS'][i-1], config['LAYERS'][i]))
        
        layers.append(nn.Linear(config['LAYERS'][-1], config['ACTION_SIZE']))

        self.layers = nn.ModuleList(layers)
        self.activation_function = config['ACTIVATION_FUNCTION']

    def forward(self, state):
        for layer in self.layers[:-1]:
            state = self.activation_function(layer(state))

        return self.layers[-1](state)

class DQNAgent:
    def __init__(self,config=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config is None:
            sys.exit("Error: No configuration provided for DQNAgent")

        self.state_size = config['STATE_SIZE']
        self.action_size = config['ACTION_SIZE']
        self.discount_factor = config['DISCOUNT_FACTOR']
        self.learning_rate = config['lr']

        self.q_network = QNetwork(config=config).to(self.device)
        self.target_q_network = QNetwork(config=config).to(self.device)
 
        self.update_target_q_network()
        self.target_q_network.eval()

        # Initialize optimizer
        self.optimizer = optim.SGD(self.q_network.parameters(), lr=self.learning_rate)

        # Initialize loss function
        self.loss_function = nn.MSELoss()

    def evaluate_mode(self):
        self.q_network.eval()
    
    def train_mode(self):
        self.q_network.train()

    def get_network_weights(self):
        return self.q_network.state_dict()
    
    def set_network_weights(self,weights):
        self.q_network.load_state_dict(weights)

    def get_learning_rate(self):
        return self.learning_rate
    
    def set_learning_rate(self,learning_rate):
        self.learning_rate = learning_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state, exploration_prob=0):
        if np.random.rand() < exploration_prob:
            return np.random.choice(self.action_size)  # Explore
        else:
            state_tensor = torch.tensor(state,dtype=torch.float,device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            
            return torch.argmax(q_values).item()  # Exploit
              
    def replay(self,replay_buffer,batch_size=32,target_network=True):
    
        if replay_buffer.buffer_size() < batch_size:
            return

        # Sample a batch of experiences from the replay buffer
        states, actions, rewards, next_states = replay_buffer.sample(batch_size)

        # Convert to tensors
        states_tensor = torch.tensor(states,dtype=torch.float,device=self.device)
        actions_tensor = torch.tensor(actions,dtype=torch.long,device=self.device).view(-1, 1)
        rewards_tensor = torch.tensor(rewards,dtype=torch.float,device=self.device).view(-1, 1)
        next_states_tensor = torch.tensor(next_states,dtype=torch.float,device=self.device)

        # Q-values for the current state-action pairs
        q_values = self.q_network(states_tensor).gather(1, actions_tensor)

        # Q-values for the next states (target Q-network)
        if target_network == True:
            with torch.no_grad():
                next_q_values = self.target_q_network(next_states_tensor).max(1)[0].unsqueeze(1)
        else:
            next_q_values = self.q_network(next_states_tensor).max(1)[0].unsqueeze(1)

        # Calculate target Q-values
        target_q_values = rewards_tensor + self.discount_factor * next_q_values
        
        # Update the Q-network
        self.optimizer.zero_grad()
        loss = self.loss_function(q_values, target_q_values)
        loss_r = loss.item()
        loss.backward()
        self.optimizer.step()
        return loss_r
        
import gymnasium as gym
import numpy as np
from classes_and_functions.polynomial_decay import polynomial_decay


def reward_function(state,terminated):
    if terminated:
        return -10
    else:
        return (1 - abs(state[2])) + (5 - abs(state[0]))


def train_agent(dqn_agent,replay_buffer,config,TN=True):
    
    env = gym.make(config["ENVIRONMENT"])
    state , _ = env.reset(seed=config['SEED'])    
    loss_episode = []

    while True:
        
        action = dqn_agent.select_action(state, config['ep'])
        next_state , reward, terminated ,truncated, _ = env.step(action)
        
        reward =  reward_function(next_state,terminated)
        
        replay_buffer.add((state, action, reward, next_state))
        
        loss = dqn_agent.replay(replay_buffer, batch_size=config['BATCH_SIZE'], target_network=TN)
        
        if loss is not None:
            loss_episode.append(loss)
            
        if terminated or truncated:
            break
        
        state = next_state
    
    env.close()

    mean_loss = np.mean(loss_episode) if len(loss_episode) > 0 else 0

    return  mean_loss


def evaluate_agent(dqn_agent,config,graphical=False):
      
    if graphical == True:
        env = gym.make(config["ENVIRONMENT"],render_mode='human')
    else:
        env = gym.make(config["ENVIRONMENT"])
        
    state , _ = env.reset(seed=config['SEED'])
    steps = 0
    total_reward = 0

    while True:
        action = dqn_agent.select_action(state, 0)
        next_state ,_, terminated ,truncated, _ = env.step(action)
       
        reward =  reward_function(next_state,terminated)
        
        total_reward += reward 
        steps += 1

        if terminated or truncated:
            break

        state = next_state
        

    env.close()

    return steps,total_reward

    
def train_and_eval(dqn_agent,replay_buffer,config,start_index=1,stop_index=1,TN=False):
    
    
    mean_loss_list = []
    step_list = []
    reward_list = []
    
    for i in range(start_index,stop_index+1):
        dqn_agent.train_mode()
        mean_loss = train_agent(dqn_agent,replay_buffer,config,TN=TN)
        dqn_agent.evaluate_mode()
        steps,reward = evaluate_agent(dqn_agent,config,graphical=False)

        mean_loss_list.append(mean_loss)
        step_list.append(steps)
        reward_list.append(reward)
        

        if i % config['DECAY_TIME'] == 0:
            config['ep'] = polynomial_decay(config['INITIAL_EP'],config['MIN_EP'],i,config['EPISODES'],config['POWER_EP'])
            

        if len(mean_loss_list) > 100 and np.mean(mean_loss_list[-100:]) < 0.5:
            config['lr'] = polynomial_decay(config['INITIAL_LR'],config['MIN_LR'],i,config['EPISODES'],config['POWER_LR'])
            dqn_agent.set_learning_rate(config['lr'])


        if i % config['TARGET_UPDATE'] == 0 and TN==True:
            dqn_agent.update_target_q_network()


        if i % 100 == 0:
            print(f"Episodes: {i} eb: {round(config['ep'],5)}, lr: {round(dqn_agent.get_learning_rate(),5)}, mean loss: {round(np.mean(mean_loss_list[-100:]),5)} ,Steps: {round(np.mean(step_list[-100:]),5)}, Reward: {round(np.mean(reward_list[-100:]),0)} , buffer size: {replay_buffer.buffer_size()}")

        if len(step_list) > 1000 and np.mean(step_list[-1000:]) >= 500 and np.mean(reward_list[-1000:]) >= 2800:
            print(f"Training completed at episode {i}")
            break

        if 'DEBUG' in config.keys() and config['DEBUG'] == True and i > 5290:
            print(i,dqn_agent.get_network_weights())
            
            

    return mean_loss_list,step_list,reward_list
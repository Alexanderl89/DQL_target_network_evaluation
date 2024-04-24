import pickle

def serialize_loss_step_reward(mean_loss_l,steps_l,reward_l,name,path):
    with open(f"{path}/mean_loss{name}.pkl","wb") as f:
        pickle.dump(mean_loss_l,f)
    with open(f"{path}/steps{name}.pkl","wb") as f:
        pickle.dump(steps_l,f)
    with open(f"{path}/reward{name}.pkl","wb") as f:
        pickle.dump(reward_l,f)

def deserialize_loss_step_reward(name,path):
    with open(f"{path}/mean_loss{name}.pkl","rb") as f:
        mean_loss_l = pickle.load(f)
    with open(f"{path}/steps{name}.pkl","rb") as f:
        steps_l = pickle.load(f)
    with open(f"{path}/reward{name}.pkl","rb") as f:
        reward_l = pickle.load(f)
        
    return mean_loss_l,steps_l,reward_l


def serialize_replay_buffer(replay_buffer,path):
    with open(f"{path}/replay_buffer.pkl", "wb") as f:
        pickle.dump(replay_buffer, f)

def deserialize_replay_buffer(path):
    with open(f"{path}/replay_buffer.pkl", "rb") as f:
        replay_buffer = pickle.load(f)
    
    return replay_buffer
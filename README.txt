DQL implemented in order to test and compare agents trained with and without target network in both 
offline and online learning. 

DQN_online: This nootebook contains prepaired code to test and run an agent with or without target network against an environment fron 
open AI gymnasium. 

Comp_offline: This notebook contains prepaired code to evaluate agents in offline mode

Comp_online: This notebook contains prepaired code to evaluate agents in online mode


The current implementation for either case is adjusted for the cart pole environment but can easily be changed to be used for some other environment.
The parts that need to be changed is the reward function and evaluation parts which can be found in train_and_evaluate.py and in comp_offline.ipynb.
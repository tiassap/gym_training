# gym_training
Reinforcement Learning on OpenAI Gym environment, with implementation of Deep Q-Network and Policy Gradient.

Watch the video on YouTube:
[![Video](https://i9.ytimg.com/vi/swFtTfiDwCU/mq1.jpg?sqp=CLTDz5kG&rs=AOn4CLAF0YFeet6ewLeZyWsq3vOuPilnIg)](https://youtu.be/swFtTfiDwCU)
\
_Training done on 3 environments: Cartpole-v1 (trained with Policy Gradient), Pong-v4 (DQN), and Half-Cheetah-v4 (Policy Gradient)_

Run this command line for simulation using pretrained weights. \
**Policy Gradient**
```
python run_pg --config=<config_filename>
```
**DQN**
```
python run_dqn --config=<config_filename>
```

Use argument `--train` to train or `--record` to record video output.
Update `yaml` config file to change the parameters.


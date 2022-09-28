# gym_training
Reinforcement Learning on OpenAI Gym environment, with implementation of Deep Q-Network and Policy Gradient.

![Watch the video](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2FNNet%2FbM_tyFU_Iw.png?alt=media&token=87c58bc2-0643-4cce-94bb-7dda6d95e6e5)
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
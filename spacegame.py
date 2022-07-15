import fire
import numpy as np

import gym

# env = gym.make("ALE/AirRaid-v5")


# class obs:
# def __init__(self, o):
# self.x = o[0]/1.5
# self.y = o[1]
# self.vx = o[2]
# self.vy = o[3]
# self.ang = o[4]
# self.angvel = o[5]
# self.leftlegtouch = o[6]
# self.rightlegtouch = o[7]
#


def play(policy, organism, render=False):
    env = gym.make(
        "LunarLander-v2",
        continuous=True,
        gravity=-10.0,
        enable_wind=False,
        wind_power=15.0,
        turbulence_power=1.5,
    )

    observation, info = env.reset(seed=42, return_info=True)
    env.action_space.seed(42)
    action = env.action_space.sample()

    total_reward = 0
    total_x = 0

    while True:
        observation, reward, done, info = env.step(action)
        action = policy(observation, organism)
        if render:
            env.render()

        # input()
        total_x = abs(observation[0]) + 0.75*abs(total_x)
        total_reward = reward + 1*total_reward 

        if done:
            observation, info = env.reset(return_info=True)
            env.close()
            return total_reward #- total_x
            # return reward


if __name__ == "__main__":
    rewards = play(lambda a, b: (0, 0), 1, render=True)
    print(rewards)

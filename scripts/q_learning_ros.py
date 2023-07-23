import rospy
import numpy as np
import gym

class QLearningAgent:
    def __init__(self, env, lr=0.8, y=0.95, num_episodes=2000):
        self.env = env
        self.lr = lr
        self.y = y
        self.num_episodes = num_episodes
        self.Q = np.zeros([env.observation_space.n, env.action_space.n])

    def train(self):
        for i in range(self.num_episodes):
            s = self.env.reset()
            d = False
            j = 0
            while j < 99:
                j += 1
                a = np.argmax(self.Q[s,:] + np.random.randn(1, self.env.action_space.n)*(1./(i+1)))
                s1, r, d, _ = self.env.step(a)
                self.Q[s,a] = self.Q[s,a] + self.lr*(r + self.y*np.max(self.Q[s1,:]) - self.Q[s,a])
                s = s1
                if d == True:
                    break

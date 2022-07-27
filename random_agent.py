from collections import deque
import os
import numpy as np


class RANDOMAgent:

    def __init__(self, enable_actions, environment_name, color):
        # parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.environment_name = environment_name
        self.enable_actions = enable_actions
        self.n_actions = len(self.enable_actions)
        self.exploration = None
        self.color = color


    def init_model(self):
        # print("RANDOMAgent : I have no model")
        return
        
    def update_exploration(self, num):
        # print("RANDOMAgent : I have no model")
        return

    def update_target_model(self):
        # print("RANDOMAgent : I have no model")
        return

    def select_action(self, states, epsilon):
        return np.random.choice(self.enable_actions)

    def store_experience(self, states, action, reward, states_1, terminal):
        # print("RANDOMAgent : I have no model")
        return
    
    def reset_experience(self):
        # print("RANDOMAgent : I have no model")
        return
        
    def experience_replay(self, step, score=None):
        # print("RANDOMAgent : I have no model")
        return

    def load_model(self, model_path=None):
        # print("RANDOMAgent : I have no model")
        return

    def save_model(self, num=None, simple=False):
        # print("RANDOMAgent : I have no model")
        return

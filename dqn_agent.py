from collections import deque
import os

import numpy as np
from keras.layers import Dense, Flatten
from keras.layers import Lambda, Input, Convolution2D
from keras.models import model_from_yaml, Model
import keras.callbacks
from keras.optimizers import RMSprop

import tensorflow.python.keras.backend as KTF
from keras import backend as K
import tensorflow as tf
import copy
from util import clone_model

f_log = './log'
f_model = './models'

model_filenames = {'black':'reversi_dqn_model_black.yaml', 'white':'reversi_dqn_model_white.yaml'}
weights_filenames = {'black':'reversi_dqn_model_weights_black.hdf5', 'black':'reversi_dqn_model_weights_white.hdf5'}

INITIAL_EXPLORATION = 1.0
FINAL_EXPLORATION = 0.1
EXPLORATION_STEPS = 500

lambda1 = lambda y_true, y_pred: y_pred
lambda2 = lambda y_true, y_pred: K.zeros_like(y_pred)
losses = {'loss': lambda1, 'main_output': lambda2}
# losses = {'loss': lambda y_true, y_pred: y_pred, 'main_output': lambda y_true, y_pred: K.zeros_like(y_pred)}


def loss_func(args):
    import tensorflow as tf
    # y_true, y_pred, a = args
    y_true = args[0]
    y_pred = args[1]
    error = tf.abs(y_pred - y_true)
    quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = tf.reduce_sum(0.5 * tf.square(quadratic_part) + linear_part)
    tf.summary.scalar('loss', loss)
    
    return loss


class DQNAgent:
    """
    Multi Layer Perceptron with Experience Replay
    """

    def __init__(self, enable_actions, environment_name, color, ddqn=False):
        # parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.environment_name = environment_name
        self.enable_actions = enable_actions
        self.n_actions = len(self.enable_actions)
        self.minibatch_size = 32
        self.replay_memory_size = 100
        self.learning_rate = 0.00025
        self.discount_factor = 0.99
        self.use_ddqn = ddqn
        self.exploration = INITIAL_EXPLORATION
        self.exploration_step = (INITIAL_EXPLORATION - FINAL_EXPLORATION) / EXPLORATION_STEPS
        self.color = color
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.model_name = "{}.ckpt".format(self.environment_name)

        self.old_session = KTF.get_session()
        self.session = tf.compat.v1.Session('')    # self.session = tf.Session('')
        KTF.set_session(self.session)

        # replay memory
        self.D = deque(maxlen=self.replay_memory_size)

        # variables
        self.current_loss = 0.0

    def init_model(self):
        
        state_input = Input(shape=(1, 8, 8), name='state')
        action_input = Input(shape=[None], name='action', dtype='int32')

        x = Flatten()(state_input)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        y_pred = Dense(self.n_actions, activation='linear', name='main_output')(x)

        y_true = Input(shape=(self.n_actions, ), name='y_true')
        loss_out = Lambda(loss_func, output_shape=(1, ), name='loss')([y_true, y_pred, action_input])
        self.model = Model(inputs=[state_input, action_input, y_true], outputs=[loss_out, y_pred])
        optimizer = RMSprop
        self.model.compile(loss=losses,
                           optimizer=optimizer(lr=self.learning_rate),
                           metrics=losses)

        self.target_model = copy.copy(self.model)
        
    def update_exploration(self, num):
        if self.exploration > FINAL_EXPLORATION:
            self.exploration -= self.exploration_step * num
            if self.exploration < FINAL_EXPLORATION:
                self.exploration = FINAL_EXPLORATION

    def Q_values(self, states, isTarget=False):
        model = self.target_model if isTarget else self.model
        res = model.predict({'state': np.array([states]),
                             'action': np.array([0]),
                             'y_true': np.array([[0] * self.n_actions])
                             })
        return res[1][0]  # shapeが64

    def update_target_model(self):
        self.target_model = clone_model(self.model)

    def select_action(self, states, epsilon):
        if np.random.rand() <= epsilon:
            # random
            return np.random.choice(self.enable_actions)
        else:
            # max_action Q(state, action)
            return self.enable_actions[np.argmax(self.Q_values(states))]

    def store_experience(self, states, action, reward, states_1, terminal):
        self.D.append((states, action, reward, states_1, terminal))

    def reset_experience(self):
        self.D.clear()
        self.D = deque(maxlen=self.replay_memory_size)

    def experience_replay(self, step, score=None):
        state_minibatch = []
        y_minibatch = []
        action_minibatch = []

        # sample random minibatch
        minibatch_size = min(len(self.D), self.minibatch_size)
        minibatch_indexes = np.random.randint(0, len(self.D), minibatch_size)

        for j in minibatch_indexes:
            state_j, action_j, reward_j, state_j_1, terminal = self.D[j]
            action_j_index = np.where(self.enable_actions==action_j)[0][0]
            # action_j_index = self.enable_actions.index(action_j)

            y_j = self.Q_values(state_j)

            if terminal:
                y_j[action_j_index] = reward_j
            else:
                if not self.use_ddqn:
                    v = np.max(self.Q_values(state_j_1, isTarget=True))
                else:
                    v = self.Q_values(state_j_1, isTarget=True)[action_j_index]
                y_j[action_j_index] = reward_j + self.discount_factor * v

            state_minibatch.append(state_j)
            y_minibatch.append(y_j)
            action_minibatch.append(action_j_index)
        # state_minibatch, y_minibatch, action_minibatchに全てを入れた
            
        validation_data = None
        if score != None:
            validation_data = ({'action': np.array(action_minibatch),
                                'state': np.array(state_minibatch),
                                'y_true': np.array(y_minibatch)},
                               [np.zeros([minibatch_size]),
                                np.array(y_minibatch)])

        self.model.fit({'state': np.array(state_minibatch),
                        'action': np.array(action_minibatch),
                        'y_true': np.array(y_minibatch)},
                       [np.zeros([minibatch_size]),
                        np.array(y_minibatch)],
                       batch_size=minibatch_size,
                       epochs=1,
                       verbose=0,
                       validation_data=validation_data)

        score = self.model.predict({'state': np.array(state_minibatch),
                                    'action': np.array(action_minibatch),
                                    'y_true': np.array(y_minibatch)})
        # print(score)
        # どれをlossにするか要検討
        self.current_loss = score[0] # [0]

    def load_model(self, model_path=None):

        yaml_string = open(os.path.join(f_model, model_filenames[self.color])).read()
        self.model = model_from_yaml(yaml_string)
        self.model.load_weights(os.path.join(f_model, weights_filenames[self.color]))

        optimizer = RMSprop
        self.model.compile(loss=losses,
                           optimizer=optimizer(lr=self.learning_rate),
                           metrics=losses)

        self.target_model = copy.copy(self.model)


    def save_model(self, num=None, simple=False):

        yaml_string = self.model.to_yaml()
        model_name = model_filenames[self.color]
        weight_name = weights_filenames[self.color]
        weight_name = weight_name[:weight_name.find(".")] + (str(num) if num else '') + ".hdf5"
        open(os.path.join(f_model, model_name), 'w').write(yaml_string)
        print('save weights')
        self.model.save_weights(os.path.join(f_model, weight_name))

    def end_session(self):
        KTF.set_session(self.old_session)

import copy
import gym
import numpy as np
import random
from collections import deque
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, Reshape
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop

episodes = 100000 #1000
episode_length = 1000
render_after = 3#7000

car_action_space = {
    0 : [0,0.5,0],
    1 : [-0.5,0.5,0],
    2 : [0.5,0.5,0],
    3 : [-1,0.5,0],
    4 : [1,0.5,0],
    5 : [-0.5,0.5,0.2],
    6: [0.5,0.5,0.2],
    7: [-1,0.5,0.2],
    8: [1,0.5,0.2]
}
num_actions = 9

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

class DQNAgent:
    def __init__(self):
        self.num_actions = num_actions
        self.memory = deque(maxlen=100000) #10000
        self.gamma = 0.9  # decay rate
        self.epsilon = 1.0  # exploration
        self.epsilon_decay = .99
        self.epsilon_min = 0.05
        self.learning_rate = 0.0001
        self._build_model()

    def _build_model(self):
        # Deep-Q learning Model
        model = Sequential()
        # model.add(Dense(64, input_dim=4, activation='tanh', init='he_uniform'))
        model.add(Convolution2D(64, 4, 4, border_mode='same',
                                input_shape=(96, 96, 3))
                  )
        model.add(Activation('relu'))
        model.add(Convolution2D(128, 4, 4))
        model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2))) #No Max Pool as this we want to see translation
        model.add(Dropout(0.25))
        model.add(Convolution2D(128, 4, 4))
        model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Dropout(0.25))
        model.add(Convolution2D(128, 4, 4))
        model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Dropout(0.25))

        # model.add(Reshape( (5, 8) ))
        # model.add(LSTM(8, return_sequences=True))

        # model.add(Dense(64, input_dim=96*96*3, activation='tanh', init='he_uniform'))
        # model.add(Dense(128, activation='tanh', init='he_uniform'))
        # model.add(Dense(128, activation='tanh', init='he_uniform'))


        model.add(Flatten())
        model.add(Dense(128, activation='tanh', init='he_uniform'))
        model.add(Dense(128, activation='tanh', init='he_uniform'))
        # model.add(Dense(32, activation='tanh', init='he_uniform'))
        model.add(Dense(self.num_actions, activation='linear', init='he_uniform'))
        model.compile(loss='mse',
                      optimizer=RMSprop(lr=self.learning_rate))
        self.model = model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state): #Reform returned actions
        #Random Action
        if np.random.rand() <= self.epsilon:
            return  random.randrange(self.num_actions)

        #Else predicted action
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
           # returns action

    def replay(self, batch_size):
        batchs = min(batch_size, len(self.memory))
        batchs = np.random.choice(len(self.memory), batchs)
        for i in batchs:
            state, action, reward, next_state, done = self.memory[i]
            target = reward
            if not done:
                target = reward + self.gamma * \
                            np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            action_c = action
            target_f[0][action_c] = target
            self.model.fit(state, target_f, nb_epoch=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    env = gym.make('CarRacing-v0')
    agent = DQNAgent()
    print(agent.model.summary())
    print("Agent made...")
    #agent.load("/home/exx/gatordriving/carracing_Thu-Fri.h5")
    for e in range(episodes):
        state_queue = deque(maxlen=4)
        total_reward = 0
        # print("Episode " + str(e))
        state = env.reset()
        # print state.shape
        #state = np.reshape(state, [1, 4])
        state = np.dot(state[...,:3], [0.299, 0.587, 0.114])
        state = np.reshape(state, [1, 96, 96])
        for i in range(3):
            state_queue.append(state)
        # print state_queue
        for time_t in range(episode_length): #5000
            if e%100==0:
                env.render()
            # print("Time step " + str(time_t) )
            frame =  np.reshape(np.stack(state_queue, axis=-1),[1, 96, 96, 3])
            action = agent.act(frame)
            #print(action)
            next_state, reward, done, _ = env.step(car_action_space[action])
            #next_state = np.reshape(next_state, [1, 4])
            next_state = rgb2gray(next_state)
            next_state = np.reshape(next_state, [1, 96, 96])
            old_state = copy.deepcopy(frame)
            state_queue.append(next_state)
            state_queue.popleft()
            total_reward = total_reward + reward
            #reward = -100 if done else reward
            frame = np.reshape(np.stack(state_queue, axis=-1), [1, 96, 96, 3])
            # print done
            agent.remember(old_state, action, reward, frame, done)
            if done:
                print("episode: {}/{}, score: {}, memory size: {}, e: {}"
                      .format(e, episodes, time_t,
                              len(agent.memory), agent.epsilon))
                break

        # print(total_reward)



        if e % 50 == 0:
            agent.save("/home/exx/gatordriving/carracing.h5")
        agent.replay(1000)

    print("All episodes over")


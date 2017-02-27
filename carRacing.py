# -*- coding: utf-8 -*-
import copy
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, Reshape
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop

episodes = 3000 #1000
episode_length = 250
render_after = 1500

def format_action_rand(a):
    #Convert random actions to compatible values
    #Steer, Gas, Brake    
    
    if -1 <= a[0] < -0.5:
        return(np.array([-1, 0.2, 0]))
    if -0.5 <= a[0] <= 0.5 :
        return(np.array([0, 0.2, 0]))    
    if +1 >= a[0] > 0.5:
        return(np.array([+1, 0.2, 0]))
    
    #Else
    print("Error, a is " + str(a))


def format_action_pred(a):
    #Convert predicted actions to compatible values
    #Steer, Gas, Brake
    
    if a == 0:
        return(np.array([-1, 0.2, 0]))
    if a == 1:
        return(np.array([0, 0.2, 0]))    
    if a == 2:
        return(np.array([+1, 0.2, 0]))
    #Else
    print("Error, a is " + str(a))

def convert_action(a):
    #Convert stored actions to compatible values
    #Steer, Gas, Brake    
    
    if action[0] == -1:
        return(0)
    if action[0] ==  0:
        return(1)    
    if action[0] == +1:
        return(2)
    
    #Else
    print("Error, a is " + str(action))    
    



class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=5000) #10000
        self.gamma = 0.9  # decay rate
        self.epsilon = 1.0  # exploration
        self.epsilon_decay = .99
        self.epsilon_min = 0.05
        self.learning_rate = 0.0001
        self._build_model()

    def _build_model(self):
        # Deep-Q learning Model
        model = Sequential()
        #model.add(Dense(64, input_dim=4, activation='tanh', init='he_uniform'))
        model.add(Convolution2D(64, 8, 8, border_mode='same',
                        input_shape=(96,96,3))
                 )
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Convolution2D(128, 4, 4))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Dropout(0.25))
        model.add(Convolution2D(128, 4, 4))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Dropout(0.25))

        
        #model.add(Reshape( (5, 8) ))
        #model.add(LSTM(8, return_sequences=True))
        
        #model.add(Dense(64, input_dim=96*96*3, activation='tanh', init='he_uniform'))
        #model.add(Dense(128, activation='tanh', init='he_uniform'))
        #model.add(Dense(128, activation='tanh', init='he_uniform'))
        

        model.add(Flatten())
        model.add(Dense(256, activation='tanh', init='he_uniform'))
        model.add(Dense(128, activation='tanh', init='he_uniform'))
        model.add(Dense(32, activation='tanh', init='he_uniform'))
        model.add(Dense(3, activation='linear', init='he_uniform'))
        model.compile(loss='mse',
                      optimizer=RMSprop(lr=self.learning_rate))
        self.model = model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state): #Reform returned actions
        #Random Action
        if np.random.rand() <= self.epsilon:
            a = self.env.action_space.sample()
            return(format_action_rand(a))
        
        #Else predicted action
        act_values = self.model.predict(state)
        a = np.argmax(act_values[0])
        return(format_action_pred(a))   # returns action

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
            action_c = convert_action(action)
            target_f[0][action_c] = target
            self.model.fit(state, target_f, nb_epoch=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    #env = gym.make('CartPole-v0')
    env = gym.make('CarRacing-v0')
    agent = DQNAgent(env)
    print(agent.model.summary())
    print("Agent made...")
    # agent.load("./save/cartpole-starter.h5")

    for e in range(episodes):
        total_reward = 0
        print("Episode " + str(e))
        state = env.reset()
        #state = np.reshape(state, [1, 4])
        state = np.reshape(state, [1, 96, 96, 3])
        for time_t in range(episode_length): #5000
            if e > render_after:
                env.render()
            #print("Time step " + str(time_t) )
            action = agent.act(state)
            #print(action)
            next_state, reward, done, _ = env.step(action)
            #next_state = np.reshape(next_state, [1, 4])
            next_state = np.reshape(next_state, [1, 96, 96, 3])
            
            total_reward = total_reward + reward
            #reward = -100 if done else reward
            agent.remember(state, action, reward, next_state, done)
            state = copy.deepcopy(next_state)
            if done:
                print("episode: {}/{}, score: {}, memory size: {}, e: {}"
                      .format(e, episodes, time_t,
                              len(agent.memory), agent.epsilon))
                break
                
        print(total_reward)
                           
        if e % 50 == 0:
            agent.save("./save/carracing.h5")
        agent.replay(100)
        
    print("All episodes over")

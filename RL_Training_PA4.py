import numpy as np
import gymnasium as gym
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import re
Nu = 2                  # Predict network update frequency
Nb = 100                # Batch size
Nt = 50                # Target network update frequency
beta = 0.99             # Discount factor
Nepisodes = 200         # Total episodes
alpha = 0.001           # Learning rate
buffer_size = 25000     # Replay buffer size
min_score = -110        # Minimum episode score to include

env = gym.make('MountainCar-v0', render_mode = "human")
N_ACTIONS = env.action_space.n
N_OBSERVATIONS = env.observation_space.shape[0]

def epsilon(episode, min_eps=0.1, max_eps=1.0, decay_rate=5.0):
    return min_eps + (max_eps - min_eps) * np.exp(-decay_rate * episode / Nepisodes)

def one_hot_encode(action, n_actions):
    return np.eye(n_actions)[action]

def build_dqn():
    model = Sequential([
        Dense(32, activation='relu', input_shape=(N_OBSERVATIONS + N_ACTIONS,)),
        Dense(64, activation='relu'),
        Dense(128, activation = 'relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=alpha), loss='mse')
    return model

def choose_action(state, model, epsilon):
    state_tiled = np.tile(state, (N_ACTIONS, 1))
    actions_one_hot = np.eye(N_ACTIONS)
    q_values = model.predict(
        np.concatenate([state_tiled, actions_one_hot], axis=1),
        verbose=0
    ).flatten()
    
    if np.random.random() < epsilon:
        exp_values = np.exp(q_values - np.max(q_values))
        probs = exp_values / np.sum(exp_values)
        return np.random.choice(N_ACTIONS, p=probs)
    return np.argmax(q_values)

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.states = np.zeros((capacity, state_dim))
        self.actions = np.zeros((capacity, action_dim))
        self.rewards = np.zeros(capacity)
        self.next_states = np.zeros((capacity, state_dim))
        self.terminated = np.zeros(capacity, dtype=bool)
        self.position = 0
        self.size = 0
        self.capacity = capacity    
    def add(self, state, action, reward, next_state, terminated):
        self.states[self.position] = state
        self.actions[self.position] = one_hot_encode(action, N_ACTIONS)
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.terminated[self.position] = terminated
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)   
    def sample(self, batch_size):
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.terminated[indices]
        )
def parse_state_string(s):
    if isinstance(s, str):
        try:
            s = re.sub(r'[\[\]]', '', s).replace(',', ' ')
            return np.array([float(x) for x in s.split()])
        except:
            return np.zeros(N_OBSERVATIONS)
    return s
def load_offline_data(path):
    try:
        df = pd.read_csv(path)
        valid_data = []
        
        for play_no, group in df.groupby('Play #'):
            try:
                states = group['State'].apply(parse_state_string).values
                next_states = group['Next State'].apply(parse_state_string).values
                actions = group['Action'].values.astype(int)
                rewards = group['Reward'].values.astype(float)
                terminated = group['Terminated'].values.astype(bool)
                if all(isinstance(s, np.ndarray) for s in states) and all(isinstance(ns, np.ndarray) for ns in next_states):valid_data.append((states, actions, rewards, next_states, terminated))
            except Exception as e:
                print(f"Skipping episode {play_no}: {str(e)}")
                continue        
        if not valid_data:
            print("No valid episodes found")
            return None        
        # Combine all valid episodes
        states = np.concatenate([d[0] for d in valid_data])
        actions = np.concatenate([d[1] for d in valid_data])
        rewards = np.concatenate([d[2] for d in valid_data])
        next_states = np.concatenate([d[3] for d in valid_data])
        terminated = np.concatenate([d[4] for d in valid_data])        
        return states, actions, rewards, next_states, terminated   
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None
def train_dqn(use_offline_data=True):
    buffer = ReplayBuffer(buffer_size, N_OBSERVATIONS, N_ACTIONS)    
    if use_offline_data:
        print("Loading offline data...")
        offline_data = load_offline_data('car_dataset.csv')        
        if offline_data:
            states, actions, rewards, next_states, terminated = offline_data
            for i in range(len(states)):
                buffer.add(states[i], actions[i], rewards[i], next_states[i], terminated[i])
            print(f"Loaded {buffer.size} transitions from offline data")
        else:
            print("Could not load offline data")
            use_offline_data = False    
    predict_model = build_dqn()
    target_model = build_dqn()
    target_model.set_weights(predict_model.get_weights())    
    episode_rewards = []    
    for episode in range(Nepisodes):
        state, _ = env.reset()
        total_reward = 0
        eps = epsilon(episode)        
        for step in range(200):
            action = choose_action(state, predict_model, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward            
            buffer.add(state, action, reward, next_state, done)            
            if step % Nu == 0 and buffer.size >= Nb:
                batch = buffer.sample(Nb)
                states_b, actions_b, rewards_b, next_states_b, terminated_b = batch                
                targets = rewards_b.copy()
                non_terminal = ~terminated_b                
                if np.any(non_terminal):
                    next_q = np.zeros((np.sum(non_terminal), N_ACTIONS))
                    for a in range(N_ACTIONS):
                        next_input = np.concatenate([
                            next_states_b[non_terminal],
                            np.tile(one_hot_encode(a, N_ACTIONS), (np.sum(non_terminal), 1))
                        ], axis=1)
                        next_q[:, a] = target_model.predict(next_input, verbose=0).flatten()
                    greedy_actions = np.argmax(next_q, axis=1) # Expected SARSA
                    probs = np.ones_like(next_q) * eps / N_ACTIONS
                    probs[np.arange(len(greedy_actions)), greedy_actions] += (1 - eps)
                    expected_q = np.sum(next_q * probs, axis=1)                    
                    targets[non_terminal] += beta * expected_q                
                X = np.concatenate([states_b, actions_b], axis=1) # Train
                predict_model.train_on_batch(X, targets)
            if step % Nt == 0:
                target_model.set_weights(predict_model.get_weights())            
            state = next_state
            if done:
                break        
        episode_rewards.append(total_reward)
        print(f"Episode {episode+1}/{Nepisodes}, Reward: {total_reward}, Epsilon: {eps:.2f}")    
    return predict_model, np.array(episode_rewards)
def plot_results(rewards, window=10):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Episode Reward', alpha=0.6)    
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, label=f'{window}-episode Avg')   
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_results.png')
    plt.show()

def plot_comparison(rewards_without_offline, rewards_with_offline, window=10):
    plt.figure(figsize=(12, 8))    
    plt.plot(rewards_without_offline, label='Without Offline Data', color='red', alpha=0.4)
    plt.plot(rewards_with_offline, label='With Offline Data', color='blue', alpha=0.4)  
    if len(rewards_without_offline) >= window:     # Plot for moving averages
        moving_avg_without = np.convolve(rewards_without_offline, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards_without_offline)), moving_avg_without, 
                 color='darkred', label=f'Without Offline - {window}-ep Avg')   
    if len(rewards_with_offline) >= window:
        moving_avg_with = np.convolve(rewards_with_offline, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards_with_offline)), moving_avg_with, 
                 color='darkblue', label=f'With Offline - {window}-ep Avg')    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress: With vs. Without Offline Data')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison_results.png')
    plt.show()
if __name__ == "__main__":
    print(" Training WITH offline data")
    model_with_offline, rewards_with_offline = train_dqn(use_offline_data=True)
    model_with_offline.save('mountaincar_dqn_with_offline.h5')
    plot_results(rewards_with_offline)
    print("Training WITHOUT offline data")
    model_without_offline, rewards_without_offline = train_dqn(use_offline_data=False)
    model_without_offline.save('mountaincar_dqn_without_offline.h5')
    plot_results(rewards_without_offline)
    plot_comparison(rewards_without_offline, rewards_with_offline, window=50)
    
    env.close()
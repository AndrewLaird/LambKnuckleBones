import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
from knucklebones_env import KnuckleBonesEnv


env = KnuckleBonesEnv()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Hyperparameters
num_epochs = 100
batch_size = 32
gamma = 0.99
epsilon_clip = 0.2
lr = 0.0003

# Initialize actor and critic networks
def build_networks():
    # Actor network
    actor = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(state_size,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(action_size, activation='softmax')
    ])
    # Critic network
    critic = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(state_size,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    return actor, critic

actor, critic = build_networks()

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(lr=lr)

# Function to compute the discounted rewards
def compute_discounted_rewards(rewards):
    discounted_rewards = np.zeros_like(rewards)
    running_sum = 0
    for t in reversed(range(len(rewards))):
        running_sum = running_sum * gamma + rewards[t]
        discounted_rewards[t] = running_sum
    return discounted_rewards

# Function to compute the advantage values
def compute_advantages(rewards, values):
    advantages = rewards - values
    return (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

# Main training loop
for epoch in range(num_epochs):
    # Initialize the batch data
    batch_states = []
    batch_actions = []
    batch_rewards = []
    batch_values = []
    episode_rewards = []
    
    state = env.reset()
    done = False
    while not done:
        # Collect experience
        action_probs = actor.predict(np.array([state]))[0]
        action = np.random.choice(action_size, p=action_probs)
        next_state, reward, done, _ = env.step(action)
        
        batch_states.append(state)
        batch_actions.append(action)
        batch_rewards.append(reward)
        
        episode_rewards.append(reward)
        state = next_state
    
    # Compute the final value of the last state in the episode
    final_value = critic.predict(np.array([state]))[0][0]
    batch_values = np.append(batch_values, np.zeros_like(batch_rewards))
    batch_values[-1] = final_value
    
    # Compute the discounted rewards and advantages
    discounted_rewards = compute_discounted_rewards(batch_rewards)
    advantages = compute_advantages(discounted_rewards, batch_values)
    
    # Update the actor and critic networks using PPO
    for i in range(len(batch_states) // batch_size):
        # Select a batch of data
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        states_batch = np.array(batch_states[batch_start:batch_end])
        actions_batch = np.array(batch_actions[batch_start:batch_end])
        discounted_rewards_batch = np.array(discounted_rewards[batch_start:batch_end])
        advantages_batch = np.array(advantages[batch_start:batch_end])
        
        # Compute the old and new action probabilities
        old_action_probs = actor.predict(states_batch)
       


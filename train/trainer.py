#from agents.base_agent import BaseAgent
import numpy as np

def train_agent(agent: BaseAgent, env, episodes=1, verbose=False):
    all_rewards = []

    for ep in range(episodes):
        state = env.reset()
        agent.reset()
        done = False
        total_reward = 0

        while not done:
            action, action_id = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(action_id, reward)
            state = next_state
            total_reward += reward

        all_rewards.append(total_reward)
        if verbose:
            print(f"Épisode {ep + 1}/{episodes} — Total reward: {total_reward:.2f}")

    return all_rewards
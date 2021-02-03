import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
import numpy as np
from threading import Thread, Lock
from multiprocessing import cpu_count
import copy

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"  # this line disable pop-out window
from ple.games.flappybird import FlappyBird
from ple import PLE
# default use float32 in conda env
# tf.keras.backend.set_floatx('float64')

# set visible GPU
gpu_number = 0

#set seed gpu_number
seed = 2021

gamma = 0.99
update_interval = 5
actor_lr = 0.0005
critic_lr = 0.001
save_model_episode = 100

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[gpu_number], 'GPU')
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)



CUR_EPISODE = 0

class Actor:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(actor_lr)
        self.entropy_beta = 0.01

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(64, activation='relu'),
            Dense(8, activation='relu'),
            Dense(self.action_dim, activation='softmax')
        ])
    
    def compute_loss(self, actions, logits, advantages):
        # ppt page47 update g_pi
        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        actions = tf.cast(actions, tf.int32)
        policy_loss = ce_loss(actions, logits, sample_weight=tf.stop_gradient(advantages))

        entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        entropy = entropy_loss(logits, logits)

        # ppt page48 solution to pitfall:exploration
        return policy_loss - self.entropy_beta * entropy 

    def TA_state(self, game):
        bucket_range_per_feature = {
            'next_next_pipe_bottom_y': 40,
            'next_next_pipe_dist_to_player': 512,
            'next_next_pipe_top_y': 40,
            'next_pipe_bottom_y': 20,
            'next_pipe_dist_to_player': 20,
            'next_pipe_top_y': 20,
            'player_vel': 4,
            'player_y': 16
        }
        state = copy.deepcopy(game.getGameState())
        
        state['next_next_pipe_bottom_y'] -= state['player_y']
        state['next_next_pipe_top_y'] -= state['player_y']
        state['next_pipe_bottom_y'] -= state['player_y']
        state['next_pipe_top_y'] -= state['player_y']

# =============================================================================
#         state_key = [k for k, v in sorted(state.items())]
#         for key in state_key:
#             state[key] = int(state[key] / bucket_range_per_feature[key])
# =============================================================================
            
        relative_state = list(state.values())


        # return the state in tensor type, with batch dimension
        relative_state = tf.convert_to_tensor(relative_state, dtype=tf.float32)
        relative_state = tf.expand_dims(relative_state, axis=0)
        
        return relative_state
    
    def train(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            loss = self.compute_loss(actions, logits, advantages)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

class Critic:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(critic_lr)
    
    def create_model(self):
        return tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(8, activation='relu'),
            Dense(1, activation='linear')
        ])

    def compute_loss(self, v_pred, td_targets):
        # ppt page47 update fV_pi
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

class Agent:
    def __init__(self):
        game = FlappyBird()
        env = PLE(game, fps=30, display_screen=False, rng=seed)  # game environment interface
        env.reset_game()

        self.state_dim = len(self.TA_state(game)[0])
        self.action_dim = len(env.getActionSet()) # number of actions

        self.global_actor = Actor(self.state_dim, self.action_dim)
        self.global_critic = Critic(self.state_dim)
        self.num_workers = cpu_count() # 16 for R7-5800X

    def TA_state(self, game):
        bucket_range_per_feature = {
            'next_next_pipe_bottom_y': 40,
            'next_next_pipe_dist_to_player': 512,
            'next_next_pipe_top_y': 40,
            'next_pipe_bottom_y': 20,
            'next_pipe_dist_to_player': 20,
            'next_pipe_top_y': 20,
            'player_vel': 4,
            'player_y': 16
        }
        state = copy.deepcopy(game.getGameState())
        
        state['next_next_pipe_bottom_y'] -= state['player_y']
        state['next_next_pipe_top_y'] -= state['player_y']
        state['next_pipe_bottom_y'] -= state['player_y']
        state['next_pipe_top_y'] -= state['player_y']

# =============================================================================
#         state_key = [k for k, v in sorted(state.items())]
#         for key in state_key:
#             state[key] = int(state[key] / bucket_range_per_feature[key])
# =============================================================================
            
        relative_state = list(state.values())


        # return the state in tensor type, with batch dimension
        relative_state = tf.convert_to_tensor(relative_state, dtype=tf.float32)
        relative_state = tf.expand_dims(relative_state, axis=0)
        
        return relative_state

    def train(self, max_episodes=20000):
        workers = []

        for _ in range(self.num_workers):
            game = FlappyBird()
            env = PLE(game, fps=30, display_screen=False, rng=seed)  # game environment interface
            env.reset_game()

            workers.append(WorkerAgent(game, env, self.global_actor, self.global_critic, max_episodes))
            
        for worker in workers:
            worker.start()
        
        for worker in workers:
            worker.join()

class WorkerAgent(Thread):
    def __init__(self, game, env, global_actor, global_critic, max_episodes):
        Thread.__init__(self)
        self.lock = Lock()
        self.game = game
        self.env = env
        self.state_dim = len(self.TA_state(self.game)[0])
        self.action_dim = len(self.env.getActionSet())

        self.max_episodes = max_episodes
        self.global_actor = global_actor
        self.global_critic = global_critic
        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim)

        self.actor.model.set_weights(self.global_actor.model.get_weights())
        self.critic.model.set_weights(self.global_critic.model.get_weights())

    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0
        if not done:
            cumulative = next_v_value # estimate of fVpi(t+1)

        for k in reversed(range(0, len(rewards))):
            cumulative = gamma * cumulative + rewards[k] # ppt page 47 紅字, estimate fQpi
            td_targets[k] = cumulative
        return td_targets

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def TA_state(self, game):
        bucket_range_per_feature = {
            'next_next_pipe_bottom_y': 40,
            'next_next_pipe_dist_to_player': 512,
            'next_next_pipe_top_y': 40,
            'next_pipe_bottom_y': 20,
            'next_pipe_dist_to_player': 20,
            'next_pipe_top_y': 20,
            'player_vel': 4,
            'player_y': 16
        }
        state = copy.deepcopy(game.getGameState())
        
        state['next_next_pipe_bottom_y'] -= state['player_y']
        state['next_next_pipe_top_y'] -= state['player_y']
        state['next_pipe_bottom_y'] -= state['player_y']
        state['next_pipe_top_y'] -= state['player_y']

# =============================================================================
#         state_key = [k for k, v in sorted(state.items())]
#         for key in state_key:
#             state[key] = int(state[key] / bucket_range_per_feature[key])
# =============================================================================

        relative_state = list(state.values())


        # return the state in tensor type, with batch dimension
        relative_state = tf.convert_to_tensor(relative_state, dtype=tf.float32)
        relative_state = tf.expand_dims(relative_state, axis=0)
        
        return relative_state

    def train(self):
        global CUR_EPISODE

        while CUR_EPISODE < self.max_episodes:
            state_batch = []
            action_batch = []
            reward_batch = []
            episode_reward, done = 0, False
            
            # Reset the environment
            self.env.reset_game()
            state = self.TA_state(self.game)

            
            while not done:
                probs = self.actor.model.predict(
                    np.reshape(state, [1, self.state_dim]))
                action = np.random.choice(self.action_dim,p=probs[0])
                reward = self.env.act(self.env.getActionSet()[action])
                done = self.env.game_over()

                next_state = self.TA_state(self.game)  # get next state
                state = np.reshape(state, [1, self.state_dim])
                action = np.reshape(action, [1, 1])
                next_state = np.reshape(next_state, [1, self.state_dim])
                reward = np.reshape(reward, [1, 1])
                
                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append(reward)

                if(len(state_batch) >= update_interval or done):
                    states = self.list_to_batch(state_batch)
                    actions = self.list_to_batch(action_batch)
                    rewards = self.list_to_batch(reward_batch)

                    next_v_value = self.critic.model.predict(next_state) # fVpi(t+1)
                    td_targets = self.n_step_td_target(rewards, next_v_value, done)
                    advantages = td_targets - self.critic.model.predict(states)

                    with self.lock:
                        actor_loss = self.global_actor.train(
                            states, actions, advantages)
                        critic_loss = self.global_critic.train(
                            states, td_targets)

                        self.actor.model.set_weights(
                            self.global_actor.model.get_weights())
                        self.critic.model.set_weights(
                            self.global_critic.model.get_weights())

                    state_batch = []
                    action_batch = []
                    reward_batch = []
                    # td_target_batch = []
                    # advatnage_batch = []

                episode_reward += reward[0][0]
                state = next_state[0]
            
            if CUR_EPISODE % save_model_episode == 0:
                self.global_actor.model.save("models/ep%d"%CUR_EPISODE)

            print('EP{} EpisodeReward={}'.format(CUR_EPISODE, episode_reward))
            CUR_EPISODE += 1

    def run(self):
        self.train()

def main():
    agent = Agent()
    agent.train()

if __name__ == "__main__":
    main()
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import copy
import numpy as np
actor_lr = 0.0005

gpu_number = 0
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


class Bucket(keras.layers.Layer):
    def __init__(self):
        super(Bucket, self).__init__()
        self.bucket_range_per_feature = {
            'next_next_pipe_bottom_y': 40,
            'next_next_pipe_dist_to_player': 512,
            'next_next_pipe_top_y': 40,
            'next_pipe_bottom_y': 20,
            'next_pipe_dist_to_player': 20,
            'next_pipe_top_y': 20,
            'player_vel': 4,
            'player_y': 16
        }
        self.bucket_list = tf.constant([16,4,20,20,20,512,40,40])

    def call(self, inputs):
        

        inputs = tf.cast(tf.cast(tf.math.divide(inputs, tf.cast(self.bucket_list, tf.float32)),tf.int32),tf.float32)
        # print(inputs.values())

        return inputs


class Actor_save(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Actor_save, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.temp_model = self.create_temp_model()
        inf_model = tf.keras.models.load_model("C:\\Users\\User\\Desktop\\comp4\\hand_bucket_model\\ep1100", compile=False)
        self.mid_model = self.create_model()
        self.mid_model.set_weights(inf_model.get_weights())
        self.model = self.final_model()
        self.opt = tf.keras.optimizers.Adam(actor_lr)
        self.entropy_beta = 0.01


    def create_temp_model(self):
        input = Input(shape=(self.state_dim,))
        output = Bucket()(input)
        model = Model(inputs=input,outputs=output)
        return model
        # return tf.keras.Sequential([
        #     Input((self.state_dim,)),
        #     Bucket()
        # ])
    def create_model(self):
        return tf.keras.Sequential([
            Input(shape=(self.state_dim,)),
            Dense(64, activation='relu'),
            Dense(8, activation='relu'),
            Dense(self.action_dim, activation='softmax')
        ])
    
    def final_model(self):
        input = Input(shape=(self.state_dim))
        x = self.temp_model(input)
        output = self.mid_model(x)
        model = Model(input,output)
        # return tf.keras.Sequential([
        #         Input((self.state_dim,)),
        #         self.temp_model(),
        #         self.mid_model()
        # ])
        return model

    def compute_loss(self, actions, logits, advantages):
        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        actions = tf.cast(actions, tf.int32)
        policy_loss = ce_loss(actions, logits, sample_weight=tf.stop_gradient(advantages))

        entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        entropy = entropy_loss(logits, logits)

        # ppt page48 solution to pitfall:exploration
        return policy_loss - self.entropy_beta * entropy 
    def call(self, states):
        x = self.model(states)
        return x
    
    def TA_state(self, game):
        state = copy.deepcopy(game.getGameState())
        
        state['next_next_pipe_bottom_y'] -= state['player_y']
        state['next_next_pipe_top_y'] -= state['player_y']
        state['next_pipe_bottom_y'] -= state['player_y']
        state['next_pipe_top_y'] -= state['player_y']
            
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

actor_save = Actor_save(8,2)
input_array = tf.random.uniform((1,8))
input = np.reshape(input_array, [1, 8])
out = actor_save(input_array)
actor_save.save("submit/DL_comp4_16_model")
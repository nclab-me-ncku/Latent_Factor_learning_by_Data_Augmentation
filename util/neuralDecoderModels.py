import tensorflow as tf
from tensorflow import keras

class Encoder(keras.Model):
    def __init__(self):
        super().__init__()

        self.projector = keras.Sequential([ 
            keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)),
            keras.layers.LSTM(256, return_sequences=False),
        ])        
            
    def call(self, inputs):        
        y = self.projector(inputs)        
        return y
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 256)

class neural_decoder(keras.Model):
    def __init__(self, output_dim=1):
        super().__init__()

        self.layer = keras.Sequential([ 
            keras.layers.Flatten(),
            keras.layers.Activation('tanh'),
            keras.layers.Dense(64, activation='tanh'),
            keras.layers.Dense(output_dim),
        ])        
            
    def call(self, inputs):        
        y = self.layer(inputs)        
        return y

class LSTM_neuralDecoder(keras.Model):
    def __init__(self, output_dim=1):
        super().__init__()

        self.encoder = Encoder()
        
        self.decoder = neural_decoder(output_dim=output_dim)     
            
    def call(self, inputs):
        x = self.encoder(inputs)   
        y = self.decoder(x)        
        return y

class extractor(keras.Model):
    def __init__(self, output_dim=1):
        super().__init__()


        self.encoder = Encoder()
        self.neural_decoder = neural_decoder(output_dim=output_dim)
        self.predictor = keras.Sequential([
            keras.layers.Dense(200, use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('tanh'),
            keras.layers.Dense(256),
        ])

        self.loss_tracker_domain = keras.metrics.Mean(name='domain_loss')
        self.loss_tracker_decode = keras.metrics.Mean(name='decode_loss')

    def train_step(self, data):
        data = data[0]

        view_1 = data['x1']
        view_2 = data['x2']
        kinematic = data['y']

        with tf.GradientTape() as tape, tf.GradientTape() as decode_tape:
            # Forward pass
            z1 = self.encoder(view_1, training=True)
            z2 = self.encoder(view_2, training=True)

            p1 = self.predictor(z1, training=True)
            p2 = self.predictor(z2, training=True)            

            # Compute our own loss
            re_loss_1 = tf.reduce_mean(keras.losses.cosine_similarity(tf.stop_gradient(z1), p2), axis=-1)
            re_loss_2 = tf.reduce_mean(keras.losses.cosine_similarity(tf.stop_gradient(z2), p1), axis=-1)
            re_loss = (re_loss_1 + re_loss_2) * 0.5
            
            k1 = self.neural_decoder(self.encoder(view_1))
            k2 = self.neural_decoder(self.encoder(view_2))

            decode_loss_1 = keras.losses.mean_squared_error(k1, kinematic)
            decode_loss_2 = keras.losses.mean_squared_error(k2, kinematic)
            decode_loss = (decode_loss_1 + decode_loss_2) * 0.5

        # Compute gradients        
        gradients = tape.gradient(re_loss, self.encoder.trainable_variables + self.predictor.trainable_variables)
        decode_gradients = decode_tape.gradient(decode_loss, self.encoder.trainable_variables + self.neural_decoder.trainable_variables)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables + self.predictor.trainable_variables))
        self.optimizer.apply_gradients(zip(decode_gradients, self.encoder.trainable_variables + self.neural_decoder.trainable_variables))

        self.loss_tracker_domain.update_state(re_loss)
        self.loss_tracker_decode.update_state(decode_loss)

        return {            
            'domain_loss': self.loss_tracker_domain.result(),
            'decode_loss': self.loss_tracker_decode.result(),        
        }

    def call(self, inputs):    
        feature = self.encoder(inputs)
        x = self.predictor(feature)
        return x
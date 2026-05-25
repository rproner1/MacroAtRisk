from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.ops import log_sigmoid, maximum, sigmoid, minimum, exp, ones_like, tanh, all
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

class CustomLSTMCell(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = (self.units, self.units)  # (hidden state, cell state)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        # Weights for input and hidden state
        self.kernel = self.add_weight(shape=(input_dim, 4 * self.units),
                                 initializer='glorot_uniform',
                                 name='W')
        self.recurrent_kernel = self.add_weight(shape=(self.units, 4 * self.units),
                                 initializer='orthogonal',
                                 name='U')
        self.biases = self.add_weight(shape=(4 * self.units,),
                                 initializer='zeros',
                                 name='b')
        self.built = True
    
    def call(self, inputs, states):
        h_prev, c_prev = states 

        z = tf.matmul(inputs, self.kernel) + tf.matmul(h_prev, self.recurrent_kernel) + self.biases

        i, f, c_candidate, o = tf.split(z, num_or_size_splits=4, axis=1)

        i = tf.sigmoid(i) 
        f = tf.sigmoid(f)
        c_candidate = tf.tanh(c_candidate)
        o = tf.sigmoid(o)
        c = f * c_prev + i * c_candidate
        h = o * tf.tanh(c)
        
        return h, [h, c]


class sLSTMCell(Layer):
    def __init__(
            self, 
            units, 
            kernel_initializer="glorot_uniform", 
            kernel_regularizer=None,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = (self.units, self.units, self.units, self.units)

        self.kernel_initializer = initializers.get(kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        # Weights for input and hidden state
        self.kernel = self.add_weight(shape=(input_dim, 4 * self.units),
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 name='kernel')
        self.recurrent_kernel = self.add_weight(shape=(self.units, 4 * self.units),
                                 initializer='orthogonal',
                                 name='recurrent_kernel')
        self.biases = self.add_weight(shape=(4 * self.units,),
                                 initializer='zeros',
                                 name='biases')
        self.built = True
    
    def call(self, inputs, states):
        h_prev, c_prev, n_prev, m_prev = states 

        z = tf.matmul(inputs, self.kernel) + tf.matmul(h_prev, self.recurrent_kernel) + self.biases

        i_candidate, f_candidate, z_candidate, o_candidate = tf.split(z, num_or_size_splits=4, axis=1)

        logfplusm = log_sigmoid(f_candidate) + m_prev

        m = tf.cond(
            tf.reduce_all(tf.equal(n_prev, 0.0)),
            lambda: i_candidate,
            lambda: tf.maximum(i_candidate, logfplusm),
        )
        
        # usual output gate and cell state candidate
        o = sigmoid(o_candidate)
        z = tanh(z_candidate)

        # Stabilized input and foreget gates
        i = minimum(exp(i_candidate - m), ones_like(i_candidate))
        f = minimum(exp(logfplusm - m), ones_like(i_candidate))
        
        c = f * c_prev + i * z
        n = f * n_prev + i
        h = o * c / n
        
        return h, [h, c, n, m]
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "kernel_initializer": self.kernel_initializer,
                "kernel_regularizer": self.kernel_regularizer
            }
        )
        return config
    
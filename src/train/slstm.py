from tensorflow.keras.layers import Layer, RNN
import tensorflow as tf

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
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = (self.units, self.units, self.units)  # (hidden state, cell state, stabilizer state)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        # Weights for input and hidden state
        self.kernel = self.add_weight(shape=(input_dim, 4 * self.units),
                                 initializer='glorot_uniform',
                                 name='kernel')
        self.recurrent_kernel = self.add_weight(shape=(self.units, 4 * self.units),
                                 initializer='orthogonal',
                                 name='recurrent_kernel')
        self.biases = self.add_weight(shape=(4 * self.units,),
                                 initializer='zeros',
                                 name='biases')
        self.built = True
    
    def call(self, inputs, states):
        h_prev, c_prev, m_prev = states 

        z = tf.matmul(inputs, self.kernel) + tf.matmul(h_prev, self.recurrent_kernel) + self.biases

        i_candidate, f_candidate, z_candidate, o_candidate = tf.split(z, num_or_size_splits=4, axis=1)

        m = tf.maximum(f_candidate + m_prev, i_candidate)
        
        # usual output gate and cell state candidate
        o = tf.sigmoid(o_candidate)
        z = tf.tanh(z_candidate)

        # Stabilized input and foreget gates
        i = tf.exp(i_candidate - m) 
        f = tf.sigmoid(f_candidate + m_prev - m)
        
        c = f * c_prev + i * z
        h = o * tf.tanh(c)
        
        return h, [h, c, m]
    
class sLSTM(RNN):
    def __init__(self, units, return_sequences=False, return_state=False, **kwargs):
        # 1. Instantiate your custom cell
        cell = sLSTMCell(units)
        
        # 2. Call the parent RNN constructor with the custom cell
        super().__init__(
            cell, 
            return_sequences=return_sequences, 
            return_state=return_state, 
            **kwargs
        )
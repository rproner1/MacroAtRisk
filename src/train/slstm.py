import keras

class CustomLSTMCell(keras.layers.Layer):
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

        z = keras.ops.matmul(inputs, self.kernel) + keras.ops.matmul(h_prev, self.recurrent_kernel) + self.biases

        i, f, c_candidate, o = keras.ops.split(z, indices_or_sections=4, axis=1)

        i = keras.ops.sigmoid(i) 
        f = keras.ops.sigmoid(f)
        c_candidate = keras.ops.tanh(c_candidate)
        o = keras.ops.sigmoid(o)
        c = f * c_prev + i * c_candidate
        h = o * keras.ops.tanh(c)
        
        return h, [h, c]

class LayerNormLSTMCell(keras.layers.Layer):
    def __init__(
            self, 
            units,
            kernel_initializer='glorot_uniform',
            kernel_regularizer=None,
            recurrent_regularizer=None,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = (self.units, self.units)  # (hidden state, cell state)
        self.layer_norm = keras.layers.LayerNormalization()
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = keras.regularizers.get(
            recurrent_regularizer
        )

    def build(self, input_shape):
        input_dim = input_shape[-1]
        # Weights for input and hidden state
        self.kernel = self.add_weight(
            shape=(input_dim, 4 * self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name='W'
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, 4 * self.units),
            initializer='orthogonal',
            regularizer=self.recurrent_regularizer,
            name='R'
        )
        self.biases = self.add_weight(shape=(4 * self.units,),
                                 initializer='zeros',
                                 name='b')
        
        self.layer_norm.build((None, 4 * self.units))

        self.built = True

    def call(self, inputs, states):
        h_prev, c_prev = states

        z = keras.ops.matmul(inputs, self.kernel) + keras.ops.matmul(h_prev, self.recurrent_kernel) + self.biases

        # Apply layer normalization before activation
        z = self.layer_norm(z)

        i, f, c_candidate, o = keras.ops.split(
            z, 
            indices_or_sections=4, 
            axis=1
        )

        i = keras.ops.sigmoid(i) 
        f = keras.ops.sigmoid(f)
        c_candidate = keras.ops.tanh(c_candidate)
        o = keras.ops.sigmoid(o)
        c = f * c_prev + i * c_candidate
        h = o * keras.ops.tanh(c)
        
        return h, [h, c]
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
                "recurrent_regularizer": keras.regularizers.serialize(
                    self.recurrent_regularizer
                ),
            }
        )
        return config
    

class sLSTMCell(keras.layers.Layer):
    def __init__(
            self, 
            units, 
            kernel_initializer="glorot_uniform", 
            kernel_regularizer=None,
            recurrent_regularizer=None,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = (self.units, self.units, self.units, self.units)

        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = keras.regularizers.get(recurrent_regularizer)

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

        z = keras.ops.matmul(inputs, self.kernel) + keras.ops.matmul(h_prev, self.recurrent_kernel) + self.biases

        i_candidate, f_candidate, z_candidate, o_candidate = keras.ops.split(z, indices_or_sections=4, axis=1)

        logfplusm = keras.ops.log_sigmoid(f_candidate) + m_prev

        m = keras.ops.cond(
            keras.ops.all(keras.ops.equal(n_prev, 0.0)),
            lambda: i_candidate,
            lambda: keras.ops.maximum(i_candidate, logfplusm),
        )
        
        # usual output gate and cell state candidate
        o = keras.ops.sigmoid(o_candidate)
        z = keras.ops.tanh(z_candidate)

        # Stabilized input and foreget gates
        i = keras.ops.minimum(
            keras.ops.exp(i_candidate - m), 
            keras.ops.ones_like(i_candidate)
        )
        f = keras.ops.minimum(
            keras.ops.exp(logfplusm - m), 
            keras.ops.ones_like(i_candidate)
        )
        
        c = f * c_prev + i * z
        n = f * n_prev + i
        h = o * c / n
        
        return h, [h, c, n, m]
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
                "recurrent_regularizer": keras.regularizers.serialize(
                    self.recurrent_regularizer
                ),
            }
        )
        return config
    
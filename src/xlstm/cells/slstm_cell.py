import keras

from ..layers import BlockDiagonal

@keras.saving.register_keras_serializable()   
class MultiHeadSLSTMCell(keras.layers.Layer):
    def __init__(
            self, 
            units, 
            num_heads,
            kernel_initializer="glorot_uniform",
            recurrent_initializer='orthogonal', 
            recurrent_regularizer=None,
            kernel_regularizer=None,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.head_size = self.units // self.num_heads
        self.state_size = (self.units, self.units, self.units, self.units)
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        
        self.input_block = BlockDiagonal(
            num_heads, 
            4 * self.head_size, 
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            use_bias=False
        )
        self.recurrent_block = BlockDiagonal(
            num_heads, 
            4 * self.head_size,
            kernel_initializer=recurrent_initializer,
            kernel_regularizer=recurrent_regularizer,
            use_bias=False
        )


    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.input_block.build(input_shape) # Build blocks for model.summary()

        recurrent_shape = (None, self.units)
        self.recurrent_block.build(recurrent_shape)

        # Initialize bias. Weights are initialized within BlockDiagonal
        self.biases = self.add_weight(
            shape=(self.num_heads, 4 * self.head_size),
            initializer='zeros',
            name='biases'
        )

        # sets self.built=True so build is not called on every forward pass
        super().build(input_shape) 
    
    def call(self, inputs, states):
        
        # Each state is shape (units)
        h_prev, c_prev, n_prev, m_prev = states 

        # preactivation. returns (batch, num_heads, 4 * head_size)
        z = (
            self.input_block(inputs) + 
            self.recurrent_block(h_prev) + 
            self.biases
        )

        # Reshape each state to (num_heads, head_size)
        h_prev =  keras.ops.reshape(
            h_prev, 
            (-1, self.num_heads, self.head_size),
        )
        c_prev =  keras.ops.reshape(
            c_prev, 
            (-1, self.num_heads, self.head_size),
        )
        n_prev =  keras.ops.reshape(
            n_prev, 
            (-1, self.num_heads, self.head_size)
        )
        m_prev =  keras.ops.reshape(
            m_prev, 
            (-1, self.num_heads, self.head_size)
        )

        # Splits pre-activation into one tensor of shape
        #  (batch, num_heads, head_size) for each gate
        i_candidate, f_candidate, z_candidate, o_candidate = keras.ops.split(
            z, 
            indices_or_sections=4, 
            axis=2
        )

        # Main sLSTM math
        logfplusm =  keras.ops.log_sigmoid(f_candidate) + m_prev

        m = keras.ops.cond(
            keras.ops.all(
                keras.ops.equal(n_prev, 0.0)
            ),
            lambda: i_candidate,
            lambda: keras.ops.maximum(i_candidate, logfplusm),
        )
        
        # usual output gate and cell state candidate
        o =  keras.ops.sigmoid(o_candidate)
        z =  keras.ops.tanh(z_candidate)

        # Stabilized input and foreget gates
        i = keras.ops.minimum( 
            keras.ops.exp(i_candidate - m),  
            keras.ops.ones_like(i_candidate)
        )
        f =  keras.ops.minimum( 
            keras.ops.exp(logfplusm - m),  
            keras.ops.ones_like(i_candidate)
        )
        
        c = f * c_prev + i * z
        n = f * n_prev + i
        h = o * c / n

        # Reshape states back to original shape (batch, units)
        h =  keras.ops.reshape(
            h, 
            (-1, self.num_heads*self.head_size) 
        )
        c =  keras.ops.reshape(
            c, 
            (-1, self.num_heads*self.head_size)
        )
        n =  keras.ops.reshape(
            n, 
            (-1, self.num_heads*self.head_size) 
        )
        m =  keras.ops.reshape(
            m, 
            (-1, self.num_heads*self.head_size)
        )
        
        return h, [h, c, n, m]
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'units': self.units,
                'num_heads': self.num_heads,
                'kernel_initializer': self.kernel_initializer,
                'recurrent_initializer': self.recurrent_initializer,
                'kernel_regularizer': self.kernel_regularizer,
                'recurrent_regularizer': self.recurrent_regularizer,
            }
        )
        return config
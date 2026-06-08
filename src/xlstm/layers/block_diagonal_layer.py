import keras
import math

@keras.saving.register_keras_serializable()
class BlockDiagonal(keras.layers.Layer):

    """
    Block diagonal layer that takes an entire feature vector of dimension 
    input_dim and splits it into num_heads each with dimension 
    head_dim = input_dim // num_heads. Then, each head performs a linear
    projection along the head_dim outputing a tensor of shape 
    (batch, num_heads, head_dim) 
    """

    def __init__(
            self, 
            num_heads,
            units,
            kernel_initializer='glorot_uniform',
            kernel_regularizer=None, 
            use_bias=False,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.use_bias = use_bias

    def build(self, input_shape):
        input_dim = input_shape[-1]

        if self.units % self.num_heads != 0:
            raise ValueError(
                (
                    f'The hidden units {self.units} is not divisible' 
                    f'by the number of heads {self.num_heads}.'
                 )
            )
        
        self.padded_features = (
            math.ceil(input_dim / self.num_heads) 
            * self.num_heads
            )
        self.pad_size = self.padded_features - input_dim

        # if self.pad_size != 0:
        #     print(
        #         f"""
        #         Number of features is not divisible by the number of heads.
        #         Padding features with {self.pad_size} zeros.
        #         """
        #     )

        self.padded_input_dim = input_dim + self.pad_size

        self.head_dim = self.padded_input_dim // self.num_heads

        self.W = self.add_weight( 
            shape=(self.num_heads, self.units, self.head_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name='weights'
        )

        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.num_heads, self.units)
            )
        else:
            self.b = None

    def call(self, x):
        # x comes in the shape (batch, input_dim) but we want (batch, num_heads, head_dim)
        shape = keras.ops.shape(x)
        batch = shape[0]
        features = shape[1]

        if self.pad_size > 0:
            # pad the feature dimension
            x = keras.ops.pad(x, [[0, 0], [0, self.pad_size]])
        
        
        # Partition the input
        # Note this requires that the number of features is evenly divisible 
        # by the number of heads
        x =  keras.ops.reshape(
            x, 
            (batch, self.num_heads, self.head_dim)
        )
        
        # Einstein sum as an efficient alternative to Block diagonal matrix
        #  multiplication. Equivalent to multiplying each weight matrix by the
        #  corresponding input subset (along dimension j)
        # Doing this for the whole batch results in shape 
        # (batch, num_heads, units). Bias broadcasts across batch
        y = keras.ops.einsum("nuh,bnh->bnu", self.W, x)
        if self.use_bias:
            y = y + self.b

        return y
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'num_heads': self.num_heads,
                'units': self.units,
                'use_bias': self.use_bias,
                'kernel_initializer': self.kernel_initializer,
                'kernel_regularizer': self.kernel_regularizer
            }
        )
        return config
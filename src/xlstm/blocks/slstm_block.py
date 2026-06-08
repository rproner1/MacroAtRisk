import keras

from ..cells import MultiHeadSLSTMCell

@keras.saving.register_keras_serializable()
class sLSTMBlock(keras.layers.Layer):

    def __init__(
        self,
        units,
        num_heads,
        return_sequences=False,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.return_sequnces = return_sequences
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer

        self.input_proj = keras.layers.Dense(
            units=units,
            name=f'{self.name}_input_proj'
        )

        self.ln1 = keras.layers.LayerNormalization(name=f'{self.name}_ln1')
        self.slstm = keras.layers.RNN(
            MultiHeadSLSTMCell(
                units=units,
                num_heads=num_heads,
                kernel_regularizer=kernel_regularizer,
                recurrent_regularizer=recurrent_regularizer
            ),
            return_sequences=return_sequences,
            name=f'{self.name}_slstm'
        )
        self.gn = keras.layers.GroupNormalization(
            groups=num_heads,
            name=f'{self.name}_gn'
        )
        self.add1 = keras.layers.Add(name=f'{self.name}_add1')

        self.ln2 = keras.layers.LayerNormalization(name=f'{self.name}_ln2')

        self.up_proj_left = keras.layers.Dense(
            units=int(4*units/3), 
            kernel_regularizer=kernel_regularizer,
            name=f'{self.name}_up_proj_left'
        )
        self.up_proj_right = keras.layers.Dense(
            units=int(4*units/3),
            kernel_regularizer=kernel_regularizer,
            name=f'{self.name}_up_proj_right'
        )

        self.gelu = keras.layers.Activation('gelu', name=f'{self.name}_gelu')
        self.mult = keras.layers.Multiply(name=f'{self.name}_mult')

        self.down_proj = keras.layers.Dense(
            units=units,
            kernel_regularizer=kernel_regularizer,
            name=f'{self.name}_down_proj'
        )
        self.add2 = keras.layers.Add(name=f'{self.name}_add2')

    def call(self, inputs):

        time_steps = inputs.shape[1]
        features = inputs.shape[2]
        batch_size = keras.ops.shape(inputs)[0]

        if not self.return_sequnces:
            # fuse time dimension into features 
            reshaped_inputs = keras.ops.reshape(
                inputs, 
                (batch_size,time_steps*features)
            )
            x_in = self.input_proj(reshaped_inputs) # (batch, units)
        else:
            x_in = self.input_proj(inputs) # (batch, time_steps, units)

        x = self.ln1(inputs)
        x = self.slstm(x) # (batch, units) or (batch, time_steps, units)
        x = self.gn(x)
        x = self.add1([x,x_in]) # (batch, units) or (batch, time_steps, units)

        y = self.ln2(x)

        y_l = self.up_proj_left(y)
        y_r = self.up_proj_right(y)
        y_r = self.gelu(y_r)
        y = self.mult([y_l,y_r]) 
        y = self.down_proj(y) # (batch, time_steps, units) if return seq

        y = self.add2([y,x]) # (batch, time_steps, units) if return seq

        return y
    
    def get_config(self):
        config = super().get_config()

        config.update(
            {
                'units': self.units,
                'num_heads': self.num_heads,
                'return_sequences': self.return_sequnces,
                'kernel_regularizer': self.kernel_regularizer,
                'recurrent_regularizer': self.recurrent_regularizer
            }
        )

        return config
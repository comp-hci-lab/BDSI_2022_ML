from __main__ import *

def make_model():

    def mean_binary_crossentropy(y, y_pred):
        return tf.reduce_mean(keras.losses.binary_crossentropy(y, y_pred))


    def dice_coef(y, y_pred, axis=(1, 2), smooth=0.01):
        """
        Sorenson Dice
        \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
        where T is ground truth mask and P is the prediction mask
        """
        prediction = keras.backend.round(y_pred)  # Round to 0 or 1
        intersection = tf.reduce_sum(y * y_pred, axis=axis)
        union = tf.reduce_sum(y + y_pred, axis=axis)
        numerator = tf.constant(2.) * intersection + smooth
        denominator = union + smooth
        coef = numerator / denominator
        return tf.reduce_mean(coef)
 
    
    def dice_coef_loss(y_true, y_pred):
        return 1 - dice_coef(y_true, y_pred)

    def soft_dice_coef(y, y_pred, axis=(1, 2), smooth=0.01):
        """
        Sorenson (Soft) Dice  - Don't round the predictions
        \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
        where T is ground truth mask and P is the prediction mask
        """

        intersection = tf.reduce_sum(y * y_pred, axis=axis)
        union = tf.reduce_sum(y + y_pred, axis=axis)
        numerator = tf.constant(2.) * intersection + smooth
        denominator = union + smooth
        coef = numerator / denominator

        return tf.reduce_mean(coef)
    
    
    ###No activations applied to last Conv2D layer.
    params_final = dict(kernel_size=(1, 1), padding="same",
                        kernel_initializer="he_uniform")

    params = dict(kernel_size=(3, 3), activation="relu",
                  padding="same", data_format="channels_last",
                  kernel_initializer="he_uniform")

    input_layer = keras.layers.Input(shape=(144, 144, 4), name="input_layer")

    encoder_1_a = keras.layers.Conv2D(FILTERS, name='encoder_1_a', **params)(input_layer)
    encoder_1_b = keras.layers.Conv2D(FILTERS, name='encoder_1_b', **params)(encoder_1_a)
    batchnorm_1 = keras.layers.BatchNormalization(name='batchnorm_1')(encoder_1_b)
    downsample_1 = keras.layers.MaxPool2D(name='downsample_1')(batchnorm_1)

    encoder_2_a = keras.layers.Conv2D(FILTERS*2, name='encoder_2_a', **params)(downsample_1)
    encoder_2_b = keras.layers.Conv2D(FILTERS*2, name='encoder_2_b', **params)(encoder_2_a)
    batchnorm_2 = keras.layers.BatchNormalization(name='batchnorm_2')(encoder_2_b)
    downsample_2 = keras.layers.MaxPool2D(name='downsample_2')(batchnorm_2)

    encoder_3_a = keras.layers.Conv2D(FILTERS*4, name='encoder_3_a', **params)(downsample_2)
    encoder_3_b = keras.layers.Conv2D(FILTERS*4, name='encoder_3_b', **params)(encoder_3_a)
    batchnorm_3 = keras.layers.BatchNormalization(name='batchnorm_3')(encoder_3_b)
    downsample_3 = keras.layers.MaxPool2D(name='downsample_3')(batchnorm_3)

    encoder_4_a = keras.layers.Conv2D(FILTERS*8, name='encoder_4_a', **params)(downsample_3)
    encoder_4_b = keras.layers.Conv2D(FILTERS*8, name='encoder_4_b', **params)(encoder_4_a)
    batchnorm_4 = keras.layers.BatchNormalization(name='batchnorm_4')(encoder_4_b)
    downsample_4 = keras.layers.MaxPool2D(name='downsample_4')(batchnorm_4)


    encoder_5_a = keras.layers.Conv2D(FILTERS*16, name='encoder_5_a', **params)(downsample_4)
    encoder_5_b = keras.layers.Conv2D(FILTERS*16, name='encoder_5_b', **params)(encoder_5_a)


    upsample_4 = keras.layers.UpSampling2D(name='upsample_4', size=(2, 2), interpolation="bilinear")(encoder_5_b)
    concat_4 = keras.layers.concatenate([upsample_4, encoder_4_b], name='concat_4')
    decoder_4_a = keras.layers.Conv2D(FILTERS*8, name='decoder_4_a', **params)(concat_4)
    decoder_4_b = keras.layers.Conv2D(FILTERS*8, name='decoder_4_b', **params)(decoder_4_a)


    upsample_3 = keras.layers.UpSampling2D(name='upsample_3', size=(2, 2), interpolation="bilinear")(decoder_4_b)
    concat_3 = keras.layers.concatenate([upsample_3, encoder_3_b], name='concat_3')
    decoder_3_a = keras.layers.Conv2D(FILTERS*4, name='decoder_3_a', **params)(concat_3)
    decoder_3_b = keras.layers.Conv2D(FILTERS*4, name='decoder_3_b', **params)(decoder_3_a)


    upsample_2 = keras.layers.UpSampling2D(name='upsample_2', size=(2, 2), interpolation="bilinear")(decoder_3_b)
    concat_2 = keras.layers.concatenate([upsample_2, encoder_2_b], name='concat_2')
    decoder_2_a = keras.layers.Conv2D(FILTERS*2, name='decoder_2_a', **params)(concat_2)
    decoder_2_b = keras.layers.Conv2D(FILTERS*2, name='decoder_2_b', **params)(decoder_2_a)


    upsample_1 = keras.layers.UpSampling2D(name='upsample_1', size=(2, 2), interpolation="bilinear")(decoder_2_b)
    concat_1 = keras.layers.concatenate([upsample_1, encoder_1_b], name='concat_1')
    decoder_1_a = keras.layers.Conv2D(FILTERS, name='decoder_1_a', **params)(concat_1)
    decoder_1_b = keras.layers.Conv2D(FILTERS, name='decoder_1_b', **params)(decoder_1_a)

    last_layer = tf.keras.layers.Conv2D(name="last_layer",
                                    filters=1, **params_final)(decoder_1_b)
    output_layer = tf.keras.layers.Activation('sigmoid')(last_layer)

    print()
    print('Input size:', input_layer.shape)
    print('Output size:', output_layer.shape)
    
    
    model = keras.models.Model(inputs=input_layer, outputs=output_layer,
                               name = 'model_' + LAYER_NAME)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                  loss=dice_coef_loss,
                  metrics=[dice_coef_loss, mean_binary_crossentropy, "accuracy", dice_coef, soft_dice_coef],
                  )

    return model
from __main__ import *

def make_model():

    class KLDivergence:
        def __init__(self, q_dist, p_dist):
            self.q_dist = q_dist
            self.p_dist = p_dist
        def call(self):
            return tfp.distributions.kl_divergence(self.q_dist, self.p_dist)

    def mean_binary_crossentropy(y, y_pred):
        return tf.reduce_mean(keras.losses.binary_crossentropy(y, y_pred))

    def sum_binary_crossentropy(y, y_pred):
        return DATA_SIZE * mean_binary_crossentropy(y, y_pred)

    def likelihood_loss(y, y_pred):
        return sum_binary_crossentropy(y, y_pred)

    posterior_fn = tfp.layers.default_mean_field_normal_fn(
              loc_initializer=tf.random_normal_initializer(
                  mean=PRIOR_MU, stddev=0.05),
              untransformed_scale_initializer=tf.random_normal_initializer(
                  mean=np.log(np.exp(0.001) - 1), stddev=0.05))

    def prior_fn(dtype, shape, name, trainable, add_variable_fn):
        dist = tfp.distributions.Normal(loc=PRIOR_MU*tf.ones(shape, dtype),
                                 scale=PRIOR_SIGMA*tf.ones(shape, dtype))
        multivar_dist = tfp.distributions.Independent(dist, reinterpreted_batch_ndims=tf.size(dist.batch_shape_tensor()))
        return multivar_dist

    flipout_params = dict(kernel_size=(3, 3), activation="relu", padding="same",
                  kernel_prior_fn=prior_fn,
                  bias_prior_fn=prior_fn,
                  kernel_posterior_fn=posterior_fn,
                  bias_posterior_fn=posterior_fn,
                  kernel_divergence_fn=None,
                  bias_divergence_fn=None)

    flipout_params_final = dict(kernel_size=(1, 1), activation="sigmoid", padding="same",
                                kernel_prior_fn=prior_fn,
                                bias_prior_fn=prior_fn,
                                kernel_posterior_fn=posterior_fn,
                                bias_posterior_fn=posterior_fn,
                                kernel_divergence_fn=None,
                                bias_divergence_fn=None)

    params_final = dict(kernel_size=(1, 1), activation="sigmoid", padding="same",
                        data_format="channels_last",
                        kernel_initializer="he_uniform")

    params = dict(kernel_size=(3, 3), activation="relu",
                  padding="same", data_format="channels_last",
                  kernel_initializer="he_uniform")


    input_layer = keras.layers.Input(shape=(144, 144, 4), name="input_layer")

    encoder_1_a = tfp.layers.Convolution2DFlipout(FILTERS, name='encoder_1_a', **flipout_params)(input_layer)
    encoder_1_b = keras.layers.Conv2D(FILTERS, name='encoder_1_b', **params)(encoder_1_a)
    downsample_1 = keras.layers.MaxPool2D(name='downsample_1')(encoder_1_b)

    encoder_2_a = keras.layers.Conv2D(FILTERS*2, name='encoder_2_a', **params)(downsample_1)
    encoder_2_b = keras.layers.Conv2D(FILTERS*2, name='encoder_2_b', **params)(encoder_2_a)
    downsample_2 = keras.layers.MaxPool2D(name='downsample_2')(encoder_2_b)

    encoder_3_a = keras.layers.Conv2D(FILTERS*4, name='encoder_3_a', **params)(downsample_2)
    encoder_3_b = keras.layers.Conv2D(FILTERS*4, name='encoder_3_b', **params)(encoder_3_a)
    downsample_3 = keras.layers.MaxPool2D(name='downsample_3')(encoder_3_b)

    encoder_4_a = keras.layers.Conv2D(FILTERS*8, name='encoder_4_a', **params)(downsample_3)
    encoder_4_b = keras.layers.Conv2D(FILTERS*8, name='encoder_4_b', **params)(encoder_4_a)
    downsample_4 = keras.layers.MaxPool2D(name='downsample_4')(encoder_4_b)


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

    output_layer = keras.layers.Conv2D(name="output_layer",
                                    filters=1, **params_final)(decoder_1_b)

    print()
    print('Input size:', input_layer.shape)
    print('Output size:', output_layer.shape)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer,
                               name = 'model_' + LAYER_NAME)

    for layer in model.layers:
        if type(layer) == tfp.python.layers.conv_variational.Conv2DFlipout:
            layer.add_loss(KLDivergence(layer.kernel_posterior, layer.kernel_prior).call)
            layer.add_loss(KLDivergence(layer.bias_posterior, layer.bias_prior).call)

    model.compile(optimizer=keras.optimizers.Nadam(learning_rate=1e-4),
                  loss=likelihood_loss,
                  metrics=[likelihood_loss, mean_binary_crossentropy],
                  )

    return model

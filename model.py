from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate


# TODO: Experiment with concatenating imgs themselves
def get_model():
    input_1 = Input(shape=(224, 224, 3))
    input_2 = Input(shape=(224, 224, 3))

    concatenated_inputs = Concatenate()([input_1, input_2])


    # Shared convolutional layers
    conv_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(concatenated_inputs)
    conv_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_2)
    flatten = Flatten()(pool_1)

    fc_1 = Dense(256, activation='relu')(flatten)
    output = Dense(1, activation='sigmoid')(fc_1)


    model = Model(inputs=[input_1, input_2], outputs=output)

    # Compile & return model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


if __name__ == "__main__":
    model = get_model()
    model.summary()
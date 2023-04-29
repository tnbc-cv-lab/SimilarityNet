from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate


# TODO: Experiment with concatenating imgs themselves
def get_model():
    input_1 = Input(shape=(224, 224, 3))
    input_2 = Input(shape=(224, 224, 3))

    # Shared convolutional layers
    conv_1 = Conv2D(32, (3, 3), activation='relu', padding='same')
    # conv_2 = Conv2D(64, (3, 3), activation='relu', padding='same')
    pool_1 = MaxPooling2D(pool_size=(2, 2))
    flatten = Flatten()

    # Apply convolutional layers to input images
    feat_1 = flatten(pool_1((conv_1(input_1))))
    feat_2 = flatten(pool_1((conv_1(input_2))))

    # Concatenate features and apply fully connected layers
    merged = concatenate([feat_1, feat_2])
    fc_1 = Dense(128, activation='relu')(merged)
    output = Dense(1, activation='sigmoid')(fc_1)


    model = Model(inputs=[input_1, input_2], outputs=output)

    # Compile & return model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
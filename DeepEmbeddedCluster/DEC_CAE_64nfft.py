def CAE_for_DEC(n1, n2, n3, mindim, k): # starting for gsi file
    # based on https://blog.keras.io/building-autoencoders-in-keras.html

    from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Flatten, Reshape, Masking
    from keras.models import Model
    from keras import backend as K
    import keras
    from keras.regularizers import l1, l2
    import numpy as np
    import sklearn
    from keras.engine.topology import Layer, InputSpec
    
        
    input_img = Input(shape=(n1, n2, n3), name='input')  # adapt this if using `channels_first` image data format
    nfilt = 8
    x = Conv2D(nfilt, (3, 3), activation='relu', strides=(2,2), padding='same')(input_img)
    x = Conv2D(nfilt, (2, 1), activation='relu', strides=(1,1))(x)
    x = Conv2D(nfilt*2, (3, 3), activation='relu', strides=(2,2), padding='same')(x)
    x = Conv2D(nfilt*2, (2, 2), activation='relu', strides=(1,1))(x)

    x = Conv2D(nfilt*4, (2, 2), activation='relu', strides=(2,2))(x)
  #  x = Conv2D(nfilt*4, (1, 2), activation='relu', strides=(1,1))(x) 
 
    x = Conv2D(nfilt*8, (1, 2), activation='selu', strides=(1,2))(x) 
    x = Flatten()(x)
    encoded = Dense(mindim, activation='relu',name='encoded')(x)
    x = Dense(nfilt*8*5*2, activation='relu')(encoded)
    x = Reshape((2,5,nfilt*8))(x)
    
    x = Conv2DTranspose(nfilt*4, (1, 2), activation='relu', strides=(1,2))(x)
    
    #x = Conv2DTranspose(nfilt*4, (1, 2), activation='relu', strides=(1,1))(x)
    x = Conv2DTranspose(nfilt*2, (2, 2), activation='relu',  strides=(2,2))(x)
    
    x = Conv2DTranspose(nfilt*2, (2, 2), activation='relu', strides=(1,1))(x)
    x = Conv2DTranspose(nfilt, (3, 3), strides=(2,2), activation='relu', padding='same')(x) 
    x = Conv2DTranspose(nfilt, (2, 1), strides=(1,1), activation='relu')(x) 
    decoded = Conv2DTranspose(n3, (3, 3), activation='linear', strides=(2,2),padding='same')(x) 
    encoder = Model(input_img, encoded)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='Adam', loss='mse')
    
    return autoencoder, encoder
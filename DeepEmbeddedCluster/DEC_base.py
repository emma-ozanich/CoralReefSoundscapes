def DEC(k, alpha, CAE, encoder): # starting for gsi file
    
    # based on https://blog.keras.io/building-autoencoders-in-keras.html
    import numpy as np
    np.random.seed(2000)
    from tensorflow import set_random_seed
    set_random_seed(2000)
    
    from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Flatten, Reshape, Masking
    from keras.models import Model
    from keras import backend as K
    import keras
    from keras.regularizers import l1, l2
    import numpy as np
    import sklearn
    from keras.engine.topology import Layer, InputSpec
    
    import tensorflow as tf
    
    # randomize centroid positions
    #def DEC_loss(input_img, decoded, encoded, k):
    #    qnk = ((1 + np.norm()**2)**(-1)) / (np.sum((1+)**(-1)))
    #    pnk = (qnk**2/np.sum(qnk)) / (np.sum( (qnk**2 / np.sum(qnk)) )
    class ClusteringLayer(Layer):
        """
        Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
        sample belonging to each cluster. The probability is calculated with student's t-distribution.
        # Example
        ```
            model.add(ClusteringLayer(n_clusters=10))
        ```
        # Arguments
            n_clusters: number of clusters.
            weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
            alpha: parameter in Student's t-distribution. Default to 1.0.
        # Input shape
            2D tensor with shape: `(n_samples, n_features)`.
        # Output shape
            2D tensor with shape: `(n_samples, n_clusters)`.
        """
        def __init__(self, n_clusters, alpha, weights=None, **kwargs): #self is place holder for futre object 
            if 'input_shape' not in kwargs and 'input_dim' in kwargs:
                kwargs['input_shape'] = (kwargs.pop('input_dim'),)
            super(ClusteringLayer, self).__init__(**kwargs)
            #initialize object attributes
            self.n_clusters = n_clusters 
            alpha1 = 1#*alpha[0]
            alpha2 = 1#*alpha[1]
            self.alpha = np.array([alpha1,alpha2])#alpha #exponent for soft assignment calculation
            self.weightedloss = alpha
            self.initial_weights = weights
            self.input_spec = InputSpec(ndim=2)

        def build(self, input_shape):
            assert len(input_shape) == 2
            input_dim = input_shape[1]
            self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
            self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
            if self.initial_weights is not None:
                self.set_weights(self.initial_weights)
                del self.initial_weights
            self.built = True

        def call(self, inputs, **kwargs):
            q = tf.divide((self.alpha + 1.0)/2.0, 1.0 + tf.divide(K.sum(K.square(
                        K.expand_dims(inputs, axis=1) - self.clusters), axis=2),  self.alpha) )
                #q **= (self.alpha[i] + 1.0) / 2.0
            #q = tf.multiply(q, self.weightedloss)
            q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) 
            return q

        def compute_output_shape(self, input_shape):
            assert input_shape and len(input_shape) == 2
            return input_shape[0], self.n_clusters

        def get_config(self):
            config = {'n_clusters': self.n_clusters}
            base_config = super(ClusteringLayer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    
    #model = Model(inputs=autoencoder.input, outputs=[clustering_layer, autoencoder.output]) #Input: Spectrograms, 
                                                                                        #Output: Cluster assignments 
                                                                                        #      & Reconstructions            

    opt = keras.optimizers.Adam(lr=0.001)
    clustering_layer = ClusteringLayer(k, alpha, name='clustering')(encoder.output)
    DEC = Model(inputs=CAE.input, outputs=[clustering_layer, CAE.output]) 
    DEC.compile(optimizer=opt,loss=['kld','mse'],loss_weights=[.1,0.9])

    
    return DEC
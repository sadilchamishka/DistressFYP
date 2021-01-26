import numpy as np
#from keras.utils import np_utils
#from keras import backend as K
#K.common.set_image_dim_ordering('th')

def preprocess(X_train, X_test):
    """
    Convert from float64 to float32 and normalize normalize to decibels
    relative to full scale (dBFS) for the 4 sec clip.
    """
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = np.array([(X - X.min()) / (X.max() - X.min()) for X in X_train])
    X_test = np.array([(X - X.min()) / (X.max() - X.min()) for X in X_test])
    return X_train, X_test


def prep_train_test(X_train, y_train, X_test, y_test, nb_classes):
    """
    Prep samples ands labels for Keras input by noramalzing and converting
    labels to a categorical representation.
    """
    print('Train on {} samples, validate on {}'.format(X_train.shape[0],
                                                       X_test.shape[0]))

    # normalize to dBfS
    X_train, X_test = preprocess(X_train, X_test)

    # Convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, X_test, Y_train, Y_test

def prep_full_test(X_train):
    """
    Prep samples ands labels for Keras input by noramalzing and converting
    labels to a categorical representation.
    """

    # normalize to dBfS
    #X_train = X_train.astype('float32')

    X_train = np.array([(X - X.min()) / (X.max() - X.min()) for X in X_train])
    
    return X_train
	
def keras_img_prep(X_train, X_test, img_dep, img_rows, img_cols):
    """
    Reshape feature matrices for Keras' expexcted input dimensions.
    For 'th' (Theano) dim_order, the model expects dimensions:
    (# channels, # images, # rows, # cols).
    """
    if K.common.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    return X_train, X_test, input_shape
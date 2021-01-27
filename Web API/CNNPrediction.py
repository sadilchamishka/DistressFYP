import tensorflow as tf
from keras.models import load_model
from FeatureExtraction import stft_matrix, get_random_samples
from FeaturePreprocess import prep_full_test, keras_img_prep

def get_predictions_from_cnn(audio):
    data = get_random_samples(stft_matrix(audio),46,125)
    data = prep_full_test(data)

    # 513x125x1 for spectrogram with crop size of 125 pixels
    img_rows, img_cols, img_depth = data.shape[1], data.shape[2], 1

    # reshape image input for Keras
    # used Theano dim_ordering (th), (# chans, # images, # rows, # cols)
    data, input_shape = keras_img_prep(data, img_depth, img_rows, img_cols)

    model = load_model('cnn_5100.h5')
    model.pop()
    model.pop()
    model.pop()
    model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
    return model.predict(data)
    



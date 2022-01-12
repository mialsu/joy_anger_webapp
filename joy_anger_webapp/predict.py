def predict(sentence):
    import pickle
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import numpy as np
    import os
   
    maxlen=50
    classes = set(['joy', 'anger', 'neutral'])
    class_to_index = dict((c,i) for i, c in enumerate(classes))
    index_to_class = dict((v,k) for k, v in class_to_index.items())

    model = tf.keras.models.load_model('joy_anger_webapp/model.kerasmodel')
    model = tf.keras.models.load_model('joy_anger_webapp/model.h5')
    tokenizer = pickle.load('tokenizer.pickle', 'rb')
    sequence = tokenizer.texts_to_sequences([sentence])
    paddedSequence = pad_sequences(sequence, truncating = 'post', padding='post', maxlen=maxlen)
    p = model.predict(np.expand_dims(paddedSequence[0], axis=0))[0]
    pred_class=index_to_class[np.argmax(p).astype('uint8')]

    if pred_class == 'joy':
        return 'joy'
       
    elif pred_class == 'anger':
        return 'anger'
    
    else:
        return 'neutral'

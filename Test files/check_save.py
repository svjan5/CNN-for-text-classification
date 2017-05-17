import keras
model = keras.models.load_model('./test.h5')
model.save('hello.h5')
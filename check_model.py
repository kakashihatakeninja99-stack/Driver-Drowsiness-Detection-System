import tensorflow as tf
model = tf.keras.models.load_model('drowsy.keras')
model.summary()
print("Input shape:", model.input_shape)
import tensorflow as tf

# Convert the model

model_path = 'C:/vsc_codes/vsc_codes_py_livingLab/Trash_classifier/saved_models/MobileNetV2'
	
converter = tf.lite.TFLiteConverter.from_saved_model(model_path) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
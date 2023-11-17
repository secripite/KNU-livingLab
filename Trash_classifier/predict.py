import tensorflow as tf
import cv2
import glob
def prediction(img):
	#rescaling image
	img = img/255

	#converting to tensor
	tensor_img = tf.convert_to_tensor(img,dtype=tf.float32)

	#resizing image
	tensor_img = tf.image.resize(tensor_img,[224,224])
	tensor_img = tensor_img[tf.newaxis,...,]

	class_names = ['glass','metal','paper', 'paper2','plastic','trash']

	#predicting image
	return class_names[model.predict(tensor_img).argmax()]


if __name__ == '__main__':


	#loading model net
	model_path = 'C:/vsc_codes/vsc_codes_py_livingLab/Trash_classifier/saved_models/MobileNetV2'
	model = tf.keras.models.load_model(model_path)	

	#loading image
	image_paths = glob.glob('C:/vsc_codes/vsc_codes_py_livingLab/Trash_classifier/prediction_image/*')

	#predicting image
	for image_path in image_paths:
		img = cv2.imread(image_path)
		print('prediction for {} is :'.format(image_path.split('/')[-1]),end=' ')
		print(prediction(img))

	
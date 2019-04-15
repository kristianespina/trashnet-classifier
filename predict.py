from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
CATEGORIES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

img = image.load_img('test.jpg', target_size=(50, 50))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

model = load_model('trashnet-vgg16.h5')
predicted_class = model.predict_classes(x)

print(CATEGORIES[predicted_class[0]])

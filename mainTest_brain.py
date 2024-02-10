import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumor.h5')

image=cv2.imread(r'C:\Users\clins\Desktop\CANCER_SYSTEM\datasets_brain\test\pred11.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)

result=np.argmax(model.predict(input_img), axis=-1)

#model.predict_classes(input_img)
print(result)





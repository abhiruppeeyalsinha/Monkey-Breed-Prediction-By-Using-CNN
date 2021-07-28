import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.image import img_to_array,load_img
from keras.models import load_model
import os,cv2
import numpy as np
from PIL import Image

model = load_model(r"E:\Projects & Tutorial\CNN Project\Monkey breed\Model_section\New_model-40.h5")
img_width = 224
img_height = 224
text_file = pd.read_csv("monkey_labels.txt")



test_folder = 'validation'
path = os.listdir(test_folder)
path = len(path)
for x in range(path):
    myPic_list = os.listdir(test_folder+'/'+str(x))
    for y in myPic_list:
        img_path =(test_folder+'/'+str(x)+'/'+y)   # y = stores the individual value of an Image
        display_img = Image.open(img_path)
        test_img = load_img(img_path,target_size=(img_width,img_height))
        test_img = img_to_array(test_img)
        test_img = test_img/255
        test_img = np.expand_dims(test_img,axis=0)
        result = model.predict(test_img)
        result = np.nanargmax(result)
        txt_file = text_file.iloc[result]
        Name = txt_file.iloc[2]
        plt.imshow(display_img)
        plt.title(Name, fontsize=25)
        plt.show()

import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D,Dense,Flatten
from keras.preprocessing.image import ImageDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

taget_h = 224
taget_w = 224
train_Image_datagenarator = ImageDataGenerator(rescale=1./255,shear_range=0.2,
                                         zoom_range=0.2,vertical_flip=True,
                                         horizontal_flip=True)
# training Set
training_set = train_Image_datagenarator.flow_from_directory(r"E:\Projects & Tutorial\CNN Project\Monkey breed\train",
                                                             target_size=(taget_h,taget_w),
                                                        class_mode="categorical",batch_size=16)
# test size
test_Image_datagenarator = ImageDataGenerator(1./255)
test_set = test_Image_datagenarator.flow_from_directory(r"E:\Projects & Tutorial\CNN Project\Monkey breed\validation",
                                                         target_size=(taget_h,taget_w),
                                                         class_mode="categorical",batch_size=16)
cnn = Sequential()
cnn.add(Conv2D(filters=32,padding="Same",kernel_size=(3,3),activation='relu', input_shape=[taget_h,taget_w,3]))
cnn.add(MaxPool2D(pool_size=2))
cnn.add(Conv2D(filters=32,padding="Same",kernel_size=(3,3),activation='relu'))
cnn.add(MaxPool2D(pool_size=2))
cnn.add(Flatten())
cnn.add(Dense(units=128,activation='relu'))
cnn.add(Dense(units=10,activation='softmax'))
#cnn.summary()

# Train the model--------------------------------
cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
CCN_model = cnn.fit(x=training_set,validation_data=test_set,epochs=40)
CCN_model.model.save(r"E:\Projects & Tutorial\CNN Project\Monkey breed\Model_section\New_model-40.h5")





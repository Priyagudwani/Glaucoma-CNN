import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from tqdm import tqdm

def read_and_process(list_of_images):
    X=[] #images
    Y=[] #labels
    
    for image in list_of_images:
        imagee = cv2.imread(image)
        blur = cv2.GaussianBlur(imagee,(5,5),0)
        X.append(cv2.resize((blur), (240,240), interpolation=cv2.INTER_CUBIC))
        
        
        filename_w_ext = os.path.basename(image)
        label, file_extension = os.path.splitext(filename_w_ext)
        label, number = label.split("_")
        if label=='glaucoma':
            Y.append(1)
            #print(label)
        elif label== 'healthy':
            Y.append(0)
            #print(label)
        
    return X,Y 
        
 X=np.array(X)
Y=np.array(Y)

print("x:", X.shape)
print("y:",Y.shape)



from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.20,random_state=2)

print("X_trainshape:", X_train.shape)
print("X_testshape:",X_test.shape)
print("Y_trainshape:", Y_train.shape)
print("Y_testshape:",Y_test.shape)

n_train= len(X_train)
n_test= len(X_test)

#Initialising the CNN 
classifier=Sequential()

# import BatchNormalization
from keras.layers.normalization import BatchNormalization

# Step 1 convolution 

classifier.add(Conv2D(32,(3,3),input_shape=(240,240,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))#Step 2 - Pooling 

# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a third convolutional layer
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a forth convolutional layer
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a fifth convolutional layer
#classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

 Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow(X_train, Y_train,batch_size = 32)
test_set = test_datagen.flow(X_test,Y_test,batch_size = 32)

# checkpoint

from keras.callbacks import ModelCheckpoint

filepath="weights_blur.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = classifier.fit_generator(training_set,
steps_per_epoch = n_train,
epochs = 100,
validation_data = test_set,
validation_steps = n_test,callbacks=callbacks_list)

# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
  
# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)


        
data_folder_path = "../glaucoma/dataset-new" 
files=['../glaucoma/dataset-new/{}'.format(i) for i in os.listdir(data_folder_path)]
#files = os.listdir(data_folder_path)  
#print(len(files))

X,Y = read_and_process(files)


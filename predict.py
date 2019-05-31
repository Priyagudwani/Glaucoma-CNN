import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


model= load_model('weights.best.hdf5')
model.summary()


y_pred_class = model.predict(X_test)
print(y_pred_class)


from sklearn import metrics
print(metrics.accuracy_score(Y_test, y_pred_class.round()))
cm=metrics.confusion_matrix(Y_test, y_pred_class)
print(cm)
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative','Positive']
plt.title('Versicolor or Not Versicolor Confusion Matrix - Test Data')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()

X_tst,Y_tst= read_and_process(files[800:820])


 
    
x=np.array(X_tst)
tst_datagen = ImageDataGenerator(rescale = 1./255)

text_labels=[]
plt.figure(figsize=(30,20))
i=0
for batch in tst_datagen.flow(x,batch_size=1):
    pred= model.predict(batch)
    print(pred)
    if pred > 0.5:
        text_labels.append('glaucoma')
    else:
        text_labels.append('healthy')
    plt.subplot(4 , 5, i+1)
    plt.title('This is a '+ text_labels[i])
    imgplot=plt.imshow(batch[0])
    i+=1
    if i % 20==0:
        break
plt.show()


from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

imagee = cv2.imread('../h3.png')
       
#ROI
        
imgg = cv2.resize(imagee,(800,600))
img_s= imgg.copy()
#print(imgg.shape)
        
img_gray = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))   #adaptive histogram
cl1 = clahe.apply(img_gray)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(cl1)
        

rows,cols = maxLoc

x1=rows-115
y1=cols-115
x2=rows+125
y2=cols+125

crop_img = img_s[y1:y2, x1:x2]
cv2.imshow("optic disc", crop_img)
        
test_image = image.img_to_array(crop_img)
test_image = np.expand_dims(test_image, axis = 0)

model= load_model('weights.best.hdf5')
result = model.predict(test_image)


print(result)

if result==0:
    prediction = 'healthy'
else:
    prediction = 'glaucoma'

print('This is a ' + prediction)
cv2.waitKey(0)
cv2.destroyAllWindows()

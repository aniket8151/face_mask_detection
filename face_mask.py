  #import library
import numpy as np
import cv2
          
  # load data of withmask and withoutmask 

with_mask=np.load('with_mask.npy')
without_mask= np.load('without_mask.npy')
print(with_mask.shape)
print(without_mask.shape)
             
  # reshape the data 

with_mask=with_mask.reshape(200,50*50*3)
without_mask=without_mask.reshape(200,50*50*3)
print(with_mask.shape)
print(without_mask.shape)

  #load data into variable x

x=np.r_[with_mask,without_mask]
print(x.shape)

  # data withmask have 0 and without mask have 1 as value

labels=np.zeros(x.shape[0])
labels[200:]=1.0
              
#svm-support vector machine
#SVC-support vector classification
#import svm from sklearn and accuracy_score and train_test_split

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
           
 # split training data and test data into 25 and 65 ratio

x_train,x_test,y_train,y_test=train_test_split(x,labels,test_size=0.25)
print(x_train.shape)

 #pca-principal component analysis

from sklearn.decomposition import PCA
pca=PCA(n_components=3)
x_train= pca.fit_transform(x_train)
print(x_train[0])
print(x_train.shape)

names={0:'mask',1:'no mask'}

svm=SVC()
x_train,x_test,y_train,y_test=train_test_split(x,labels,test_size=0.25)
svm.fit( x_train , y_train )

#x_test=pca.transform(x_test)

y_pred=svm.predict(x_test)
print(accuracy_score(y_test,y_pred))

  #opening camera for detection mask and nomask

haar_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0)
font= cv2.FONT_HERSHEY_COMPLEX
while True:
    flag,image=capture.read()
    if flag:
        faces=haar_data.detectMultiScale(image)
        for x,y,w,h in faces:
          cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),4)
          face = image[y:y+h,x:x+w,:]
          face = cv2.resize(face,(50,50))
          face=face.reshape(1,-1)
         # face = pca.transform(face)
          pred = svm.predict(face)
          n=names[int(pred)]
          print(n)
          cv2.putText(image,n,(x,y),font,1,(240,250,250),2)
        cv2.imshow('result',image)
        if cv2.waitKey(2)==27 :
          break
capture.release()
cv2.destroyAllWindows()



# import opencv and load image 
import cv2
image=cv2.imread('R.jfif')
print(image.shape)    # checking the shape of image 
print(image[0])        # first row of the image
print("-------------------------------------------------")
print(image)
import matplotlib.pyplot as plt
#plt.imshow(image)
print("-------------------------------------------------")
      
      #showing image

while True: 
    cv2.imshow('result',image)       
    '''cv2.waitKey(0)'''
    if cv2.waitKey(2) == 27 :         #showing my image till i press escape
        break
cv2.destroyAllWindows()
print("-------------------------------------------------")
 
 # load haarcascade file 

haar_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print(haar_data.detectMultiScale(image))
while True:
   faces = haar_data.detectMultiScale(image)
   for x,y,w,h in faces:
       cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),4)
   cv2.imshow('result',image)
   if cv2.waitKey(2) == 27 :
       break 
cv2.destroyAllWindows()
print("----------------------------------------------------")

       #opening camera for collect your data  without mask

capture = cv2.VideoCapture(0)
data_withoutmask = []
while True:
    flag,image=capture.read()
    if flag:
        faces=haar_data.detectMultiScale(image)
        for x,y,w,h in faces:
          cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),4)
          face = image[y:y+h,x:x+w,:]
          face = cv2.resize(face,(50,50))
          print(len(data_withoutmask))
          if len(data_withoutmask) < 400:
              data_withoutmask.append(face)
        cv2.imshow('result',image)
        if cv2.waitKey(2)==27 or len(data_withoutmask) >=200 :
          break
capture.release()
cv2.destroyAllWindows()
print("----------------------------------------------------")
        
        # save data of without mask in numpy 

import numpy as np
np.save('without_mask.npy',data_withoutmask)
plt.imshow(data_withoutmask[0])

print("----------------------------------------------------")

       #opening camera for collect your data  with mask

capture = cv2.VideoCapture(0)
data_withmask = []
while True:
    flag,image=capture.read()
    if flag:
        faces=haar_data.detectMultiScale(image)
        for x,y,w,h in faces:
          cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),4)
          face = image[y:y+h,x:x+w,:]
          face = cv2.resize(face,(50,50))
          print(len(data_withmask))
          if len(data_withmask) < 400:
              data_withmask.append(face)
        cv2.imshow('result',image)
        if cv2.waitKey(2)==27 or len(data_withmask) >=200 :
          break
capture.release()
cv2.destroyAllWindows()
print("----------------------------------------------------")
              
        # saving data with mask in numpy 

np.save('with_mask.npy',data_withmask)
plt.imshow(data_withmask[0])



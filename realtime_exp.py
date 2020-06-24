import cv2,time
from tensorflow import keras
import pandas as pd
import numpy as np
face_cascade=cv2.CascadeClassifier("F:\\Opencv_py\haarcascade_frontalface_default.xml")

def face_crop(image):
    face=face_cascade.detectMultiScale(image,scaleFactor=1.05,minNeighbors=5)
    return face

def process_img(img):
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    x= img.astype('float')/255.0
    x = keras.preprocessing.image.img_to_array(x)
    x = np.expand_dims(x,axis=0)
    return x

model=keras.models.load_model("models/best_model.h5")
font = cv2.FONT_HERSHEY_SIMPLEX 
fontScale = 1
color = (0,0,255)
org=(250,50)
thickness = 2
exp =  ['ANGRY','DISGUST','FEAR','HAPPY','SAD','SURPRISE','NEUTRAL']
video=cv2.VideoCapture(0)
a=1
data=[]
expression=[]
while True:
    
    _, frame = video.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        img = gray[y:y+h,x:x+w]
        img= cv2.resize(img,(48,48),interpolation=cv2.INTER_AREA)
        if np.sum([img])!=0:
            img = process_img(img)
			
            preds = model.predict(img)[0]
            label=exp[preds.argmax()]
            pos= (x,y)
            data.append(img)
            expression.append(label)
            cv2.putText(frame,label,pos,cv2.FONT_HERSHEY_SIMPLEX,2,color,3)
        else:
            cv2.putText(frame,':(',org,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    cv2.imshow('Capturing....',frame)
    key=cv2.waitKey(1)
	
    if key== ord('q'):
        df=pd.DataFrame({"Image Data":data,
                            "Label":expression})
        df.to_csv("facial_data.csv")
        break

video.release()
cv2.destroyAllWindows()
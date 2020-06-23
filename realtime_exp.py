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
    a=a+1
    _,frame=video.read()
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_crop(grey)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),color=color,thickness=thickness)
        img=grey[x:x+w,y:y+h]
        try:
            if np.sum([img]) !=0:
                img=cv2.resize(img,(48,48),interpolation=cv2.INTER_AREA)
                data.append(img)
                img=process_img(img)
                class_name=model.predict_classes(img)
                label=exp[class_name[0]]
                expression.append(label)
                cv2.putText(frame,label,org, font,  
                            fontScale, color, thickness, cv2.LINE_AA)
           # else:
            #    cv2.putText(frame,"No Face :(",org, font,  
             #               fontScale, color, thickness, cv2.LINE_AA)

        except Exception as e:
            print("Broken Image!!")
       
    cv2.imshow("capturing",frame)
    key=cv2.waitKey(1)
    if key==ord("q"):
        df=pd.DataFrame({"Data":data, "Expression": expression})
        df.to_csv("facial_data.csv")
        break

video.release()
cv2.destroyAllWindows()
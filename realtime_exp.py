import cv2,time
from tensorflow import keras
import pandas as pd

face_cascade=cv2.CascadeClassifier("F:\\Opencv_py\haarcascade_frontalface_default.xml")

def face_crop(image):
    face=face_cascade.detectMultiScale(image,scaleFactor=1.05,minNeighbors=5)
    return face

def process_img(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    x=cv2.resize(img,(48,48),interpolation=cv2.INTER_AREA)
    x=x/255
    x=x.reshape((1,48,48,1))
    return x

model=keras.models.load_model("models/Model_05.h5")
font = cv2.FONT_HERSHEY_SIMPLEX 
fontScale = 1
color = (0,0,255)
org=(250,50)
thickness = 2
exp = ['ANGRY','DISGUST','FEAR','HAPPY','NEUTRAL','SAD','SURPRISE']
video=cv2.VideoCapture(0)
a=1
data=[]
expression=[]
while True:
    a=a+1
    _,frame=video.read()
    data.append(frame)
    x=process_img(frame)
    class_name=model.predict_classes(x)
    frame=cv2.putText(frame,exp[class_name[0]],org, font,  
                   fontScale, color, thickness, cv2.LINE_AA)
    #print(face_crop(img))
    #for (x,y,w,h) in face_crop(frame):
        #frame=cv2.rectangle(frame,(x,y),(x+w,y+h),color=color,thickness=thickness)
        #img=img[x:x+w,y:y+h]
    cv2.imshow("capturing",frame)
    label=exp[class_name[0]]
    expression.append(label)
    key=cv2.waitKey(1)
    if key==ord("q"):
        df=pd.DataFrame({"Data":data,
                          "Expression": expression})
        df.to_csv("facial_data.csv")
        break

video.release()
cv2.destroyAllWindows()
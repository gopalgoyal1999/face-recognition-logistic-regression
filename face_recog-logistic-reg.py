import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression

face1=np.load('gop.npy').reshape(100,50*50*3)

face2=np.load('shankar.npy').reshape(100,50*50*3)

data=np.concatenate([face1,face2])

dataset = cv2.CascadeClassifier('hr.xml')

capture = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_COMPLEX

user={0:"gopal",1:"shankar"}

labels = np.zeros((200,1))

labels[:100,:]=0.0

labels[100:,:]=1.0

logmod=LogisticRegression()

while True:
    
    ret,img = capture.read()
    
    if ret:
        
        #img=cv2.resize(img,None,fx=0.1,fy=0.1)     --for resize window
        
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        faces = dataset.detectMultiScale(gray)

        for x,y,w,h in faces:
            
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,225),2)
            
            myface=img[y:y+h,x:x+w,:]
            
            myface=cv2.resize(myface,(50,50))
            
            logmod.fit(data,labels)
            
            myface=myface.reshape(1,50*50*3)

            label=logmod.predict(myface)
            
            user_name=user[int(label)]
            
            cv2.putText(img,user_name,(x,y),font,1,(0,255,0),2)
            
        cv2.imshow('result',img)
        
        if cv2.waitKey(1)==27:
            
            break
    else:print("camera not working")
    
capture.release()

cv2.destroyAllWindows()

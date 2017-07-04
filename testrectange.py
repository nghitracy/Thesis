import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('C:\Python27\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   #Convert to gray
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        #Draw the bounder
    x,y,w,h=faces[0]
    print x
    print y
    print w
    print h
    print "faces"


    x = x + int(((w / 2) - (w * 0.6) / 2))
    w = int(w * 0.6)
    y = y + int(((h / 2) - (h * 0.6) / 2))
    h = int(h * 0.6)

    #cv2.circle(frame, (x, y), 3, 255, -1)
    #cv2.line(frame, (x,y),  (x+h, y), 125,3)
    #cv2.line(frame, (x, y), (x , y + w), 0, 3)
    #cv2.re
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = frame[y:y + h, x:x + w]
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
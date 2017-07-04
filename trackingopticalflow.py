import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('C:\Python27\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')

    # Capture frame-by-frame
while (True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   #Convert to gray
    ret1, th1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('thresh', th1)
    #cv2.imshow('bw', im_bw)
    #img1, contours, hierarchy = cv2.findContours(th1, cv2.RETR, 2)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        #Draw the bounder
    print faces
    if len(faces) == 0:
        continue

    x,y,w,h = faces[0]
    x = x + int(((w / 2) - (w * 0.6) / 2))
    w = int(w * 0.6)
    y = y + int(((h / 2) - (h * 0.6) / 2))
    h = int(h * 0.6)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = frame[y:y + h, x:x + w]
    corners = cv2.goodFeaturesToTrack(roi_gray, 100, 0.01, 10)
    corners = np.int0(corners)
    for i in corners:
        x, y = i.ravel()
        cv2.circle(roi_color, (x, y), 3, 255, -1)
    M = cv2.moments(corners)
    if (M['m00']!=0):
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx,cy=0,0
    center = cx, cy
    print center
    cv2.circle(roi_color, center, 8, 125, -1)
    cv2.imshow('frame',frame)
    break


while(True):






    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Display the resulting frame
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()

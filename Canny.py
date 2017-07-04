import cv2
import numpy as np
cap=cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('C:\Python27\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')

while(1):
    capture,img = cap.read()
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)

    # Draw the bounder
    #print faces
    if len(faces) == 0:
        continue

    x, y, w, h = faces[0]
    x = x + int(((w / 2) - (w * 0.6) / 2))
    w = int(w * 0.6)
    y = y + int(((h / 2) - (h * 0.6) / 2))
    h = int(h * 0.6)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = img_gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    corners = cv2.goodFeaturesToTrack(roi_gray, 100, 0.2, 30)
    #ret, thresh = cv2.threshold(roi_gray, 127, 255, 0)


    #cv2.circle(ROI, (5,5),10,255,-1)

    #for i in corners:
    #    x,y =i.ravel()
    #    cv2.circle(roi_color, (x, y), 3, 255, -1)
    #cv2.circle(roi_color, (x, y), 3, 255, -1)
    print


    canny = cv2.Canny(roi_color, 100, 200)
    M = cv2.moments(corners)
    if (M['m00'] != 0):
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = 0, 0
    cv2.circle(roi_color, (cx, cy), 5, 0, -1)




    #im2, contours, hierarchy = cv2.findContours(thresh, 2, 1)
    #im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #cv2.drawContours(img, approx, -1, (0, 255, 0), 3)
    #cv2.drawContours(roi_color, contours, -1, (0, 255, 0), 3)
    cv2.imshow('img',img)
    #cv2.imshow('frame',thresh)
    #cv2.imshow('roi',ROI)





    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
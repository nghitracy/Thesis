import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

browser = webdriver.Chrome()
browser.get('http://google.com/')
elem = browser.find_element_by_link_text('English')
elem.click()
elem = browser.find_element_by_link_text('About')
elem.click()
elem = browser.find_element_by_link_text('Our company')
elem.click()

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('C:\Python27\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
count = 0
list = []


def takecenter(faces):
    ox,oy,ow,oh= faces[0]
    ox = ox + int(ow * 0.2)
    ow = int(ow * 0.6)
    oy = oy + int(oh * 0.2)
    oh = int(oh * 0.6)
    cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (255, 0, 0), 2)
    cx = ox + int(ow / 2)
    cy = oy + int(oh / 2)
    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
    return cx, cy



while (True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 2)
    if len(faces) == 0:
        print "Can not detect face"
        continue
    cx,cy = takecenter(faces)
    direction="NORMAL"
    count = count + 1
    cv2.putText(frame, "dx: {}, dy: {}".format(cx, cy), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 0, 255), 1)
    cv2.putText(frame, "FACE DETECTION", (frame.shape[1]-100 , frame.shape[0]-100), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 0, 255), 1)
    print frame.shape
    if count == 20:
        count = 0
        list.append((cx, cy))
        if len(list) > 3:
            dX1 = list[-1][0] - list[-2][0]
            dX2 = list[-1][0] - list[0][0]
            dY1 = list[-1][1] - list[-2][1]
            dY2 = list[-1][1] - list[0][1]
            print "DX1",dX1
            print "DX2",dX2
            if np.abs(dX2) >= np.abs(dY2):
                if dX1 < -12 and dX2 < -12:
                    direction = "LEFT"
                elif dX1 > 12 and dX2 > 12:
                    direction = "RIGHT"
                else:
                    direction = "NORMAL"
            else:
                if dY1 > 12 and dY2 > 12:
                    direction = "DOWN"
                elif dY1 < -12 and dY2 < -12:
                    direction = "UP"
                else:
                    direction = "NORMAL"

            if direction == "LEFT":
                browser.back()
            elif direction == "RIGHT":
                browser.forward()
            elif direction == "UP":
                browser.execute_script("window.scrollBy(0,-500);")
            elif direction == "DOWN":
                browser.execute_script("window.scrollBy(0,500);")


    # Print the direction of face movement
    cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 0, 255), 3)

    cv2.imshow('new', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        # Display the resulting frame
browser.close()
cap.release()
cv2.destroyAllWindows()





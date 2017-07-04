import cv2
import numpy as np
from selenium import webdriver

# Parameters for find the goodcorners to track
feature_params = dict( maxCorners = 1,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# open the web browser
browser = webdriver.Chrome()

# auto access to Google web page
browser.get('http://google.com/')

# Capture video via the webcam
cap = cv2.VideoCapture(0)
# Loading the XML file for detect face in opencv library
face_cascade = cv2.CascadeClassifier('C:\Python27\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')

#Initial the list of x and y coordinate
listx = [0]
listy =[0]
count =0
numberofframe = 0
direction="NORMAL"
while(1):
    # Read the first frame and detect the face
    ret, frame_old = cap.read()
    frame_old = cv2.flip(frame_old, 1)
    gray_old = cv2.cvtColor(frame_old, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_old, 1.3, 5)
    # If the face can not be detected, it goes back to read the frame
    if len(faces) == 0 :
        continue

    #Drawing the rectangle to locate faces
    x,y,w,h=faces[0]
    x = x + int(((w / 2) - (w * 0.4) / 2))
    w = int(w * 0.4)
    y = y + int(((h / 2) - (h * 0.4) / 2))
    h = int(h * 0.4)
    cv2.rectangle(frame_old, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray_old[y:y + h, x:x + w]
    roi_color = frame_old[y:y + h, x:x + w]

    # Create the corner (interest point) in the region of face
    corners = cv2.goodFeaturesToTrack(roi_gray, mask = None, **feature_params)
    #corners = np.int0(corners)
    p0=corners
    cv2.imshow('fr',frame_old)
    # Convert the coordinate of corner that was detected in the region of face to the whole frame
    for i in range(len(p0)):
        p0[i][0][0] = p0[i][0][0]+ x
        p0[i][0][1] = p0[i][0][1] + y
        listx.append(p0[i][0][0])
        listy.append(p0[i][0][1])
    color = np.random.randint(0,255,(100,3))
    print p0
    break
while (1):
    # Read the next frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    numberofframe = numberofframe + 1


    #Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray_old, frame_gray, p0, None, **lk_params)

    #Calculate the difference of two point in consecutive frame
    d10= p0 - p1

    #Check if the next point is moving too fast which means the frame is interfered
    for i in d10:
        a,b = i.ravel()
        if a > 100 or b >100:
            st = None
        else:
            break

    # If the user press A keys, the new point will be created to get the better result
    k = cv2.waitKey(1) & 0xff
    if k == ord('a'):
        st = None
        print "Press a"

    # Check if the next point can be detected or not, if not, the system detects the face and create the interest point again.
    if st is None or st.all()==0 or numberofframe == 1000:
        print 'tao moi'
        faces = face_cascade.detectMultiScale(gray_old, 1.3, 5)
        if len(faces) == 0:
            continue
        x, y, w, h = faces[0]
        roi_gray = gray_old[y:y + h, x:x + w]
        p0 = cv2.goodFeaturesToTrack(roi_gray, mask=None, **feature_params)
        for i in range(len(p0)):
            p0[i][0][0] = p0[i][0][0] + x
            p0[i][0][1] = p0[i][0][1] + y
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(gray_old, frame_gray, p0, None, **lk_params)

    # If the next point is a good point
    good_new = p1[st == 1]
    good_old = p0[st == 1]


    if numberofframe == 1000:
        print "100 Frame"
        numberofframe = 0


    # The count variable is added by 1 to calculate the number of frame
    count = count + 1

    #Define the direction
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        a = np.int0(a)
        b= np.int0(b)

        # Print the coordinate on screen
        cv2.putText(frame, "dx: {}, dy: {}".format(a,b), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (0, 0, 255), 1)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        #print count

        #Check if there are 22 frames, it will define the direction of the movement
        if count == 22:
            count = 0
            listx.append(a)
            listy.append(b)
            if len(listx) > 2:
                dx = listx[-1] - listx [-2]
                dx1 = listx[-1] - listx[-3]
                dy = listy[-1] - listy [-2]
                dy1 = listy[-1] - listy [-3]
                #print listx
                #print "Vx",dx
                #print "Vx1",dx1

                if np.abs(dx) > 20  or np.abs(dy) > 20:
                    if np.abs(dx) >= np.abs(dy):
                        if np.sign(dx) == -1 and np.abs(dx1) > 22  :
                            direction = "LEFT"
                        elif np.sign(dx) == 1 and np.abs(dx1) > 22:
                            direction = "RIGHT"
                        else:
                            direction = "NORMAL"
                    else:
                        if np.sign(dy) == -1 and np.abs(dy1) > 20:
                            direction = "UP"
                        elif np.sign(dy) == 1 and np.abs(dy1) > 20:
                            direction = "DOWN"
                        else:
                            direction = "NORMAL"
                else:
                    direction = "NORMAL"
                print direction


                # For different direction, the are different in the action of web browser
                if direction == "LEFT":
                    browser.back()
                elif direction =="RIGHT":
                    browser.forward()
                elif direction == "UP":
                    browser.execute_script("window.scrollBy(0,-500);")
                elif direction == "DOWN":
                    browser.execute_script("window.scrollBy(0,500);")

        # Print the direction of face movement
        cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 3)



    cv2.imshow('frame', frame)
    # Now update the previous frame and previous points
    gray_old = frame_gray.copy()
    frame_old = frame.copy()
    p0 = good_new.reshape(-1, 1, 2)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

browser.close()
cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np

feature_params = dict( maxCorners = 1,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('C:\Python27\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
listx =[]
listy =[]
count =0
direction="NORMAL"
while(1):
    ret, frame_old = cap.read()
    frame_old = cv2.flip(frame_old, 1)
    gray_old = cv2.cvtColor(frame_old, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_old, 1.3, 5)
    cv2.imshow('fr',frame_old)
    if len(faces) == 0 :
        cv2.putText(frame_old, direction, (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 10)
        continue
    x,y,w,h=faces[0]
    x = x + int(((w / 2) - (w * 0.6) / 2))
    w = int(w * 0.6)
    y = y + int(((h / 2) - (h * 0.6) / 2))
    h = int(h * 0.6)
    cv2.rectangle(frame_old, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray_old[y:y + h, x:x + w]
    roi_color = frame_old[y:y + h, x:x + w]
    corners = cv2.goodFeaturesToTrack(roi_gray, mask = None, **feature_params)
    #corners = np.int0(corners)
    p0=corners
    #p0[0][0] = p0[0][0]+x
    #p0[0][1]= p0[0][1]+y
    #p0= np.array(p0)
    for i in range(len(p0)):
        p0[i][0][0] = p0[i][0][0]+ x
        p0[i][0][1] = p0[i][0][1] + y
        listx.append(p0[i][0][0])
        listy.append(p0[i][0][1])
    color = np.random.randint(0,255,(100,3))

    # Create a mask image for drawing purposes
    print p0
    break

while (1):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray_old, frame_gray, p0, None, **lk_params)
    if st is None or st.all()==0:
        print 'tao moi'
        faces = face_cascade.detectMultiScale(gray_old, 1.3, 5)
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
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    count = count + 1
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        a = np.int0(a)
        b= np.int0(b)
        cv2.putText(frame, "dx: {}, dy: {}".format(a,b), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (0, 0, 255), 1)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        #print count
        if count == 20:
            count = 0
            listx.append(a)
            listy.append(b)
            if len(listx) > 3:
                dx = listx[-1] - listx [-2]
                dx1 = listx[-1] - listx[-3]
                dy = listy[-1] - listy [-2]
                dy1 = listy[-1] - listy [-3]
                print listx
                print "Vx",dx
                print "Vx1",dx1

                if np.abs(dx) > 20  or np.abs(dy) > 20:
                    if np.abs(dx) >= np.abs(dy):
                        if np.sign(dx) == -1 and np.abs(dx1) > 20  :
                            direction = "LEFT"
                        elif np.sign(dx) == 1 and np.abs(dx1) > 20:
                            direction = "RIGHT"
                        else:
                            direction = "NORMAL"
                    else:
                        if np.sign(dy) == -1 and np.abs(dy1) > 22:
                            direction = "UP"
                        elif np.sign(dy) == 1 and np.abs(dy1) > 22:
                            direction = "DOWN"
                        else:
                            direction = "NORMAL"
                else:
                    direction = "NORMAL"
                print direction
        cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 3)



    cv2.imshow('frame', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    gray_old = frame_gray.copy()
    frame_old = frame.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()

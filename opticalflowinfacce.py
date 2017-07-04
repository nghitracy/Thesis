import cv2
import numpy as np
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('C:\Python27\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')

    # Capture frame-by-frame
while (True):
    ret, old_frame = cap.read()
    old_frame = cv2.flip(old_frame, 1)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)   #Convert to gray
    #ret1, th1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    #cv2.imshow('thresh', th1)
    #cv2.imshow('bw', im_bw)
    #img1, contours, hierarchy = cv2.findContours(th1, cv2.RETR, 2)
    faces = face_cascade.detectMultiScale(old_gray, 1.3, 5)
        #Draw the bounder
    print faces
    if len(faces) == 0:
        continue

    x,y,w,h = faces[0]
    x = x + int(((w / 2) - (w * 0.6) / 2))
    w = int(w * 0.6)
    y = y + int(((h / 2) - (h * 0.6) / 2))
    h = int(h * 0.6)
    cv2.rectangle(old_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = old_gray[y:y + h, x:x + w]
    roi_color = old_frame[y:y + h, x:x + w]
    p0 = cv2.goodFeaturesToTrack(roi_gray, 10, 0.01, 5)
    for i in range(len(p0)):
        p0[i][0][0] = p0[i][0][0]+ x
        p0[i][0][1] = p0[i][0][1] + y
    mask = np.zeros_like(old_frame)
    color = np.random.randint(0, 255, (100, 3))

    while (True):
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        print "p0:",p0
        print "p1", p1
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    if k == 27:
        break



    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

# Display the resulting frame
cap.release()
cv2.destroyAllWindows()

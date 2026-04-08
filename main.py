import cv2
import numpy as np 
import gesture_engine as htm
import time 
import autopy 

wCam, hCam = 640, 480
frameR = 100

plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()

pTime = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    mode = "None"
    color = (255,0,255)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()

        # ========================
        # Modes
        # ========================
        if fingers == [1,1,1,1,1]:
            mode = "Pause"
            color = (100,100,100)

        elif fingers[1] == 1 and fingers[2] == 0:
            mode = "Move"
            color = (0,255,0)

        elif fingers[1] == 1 and fingers[2] == 1:
            mode = "Click"
            color = (0,0,255)

        elif fingers[0] == 1:
            mode = "Right Click"
            color = (255,255,0)

        elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
            mode = "Scroll"
            color = (255,0,0)

        cv2.rectangle(img,(frameR, frameR),(wCam-frameR, hCam-frameR),
                        (255, 0, 255), 2)
        
        # ========================
        # Move Mode
        # ========================
        if mode == "Move":
            x3 = np.interp(x1, (frameR, wCam-frameR), (0,wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0,hScr))

            distance = np.hypot(x1 - plocX, y1 - plocY)
            smoothening = max(3, min(10, int(distance / 10)))

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            autopy.mouse.move(wScr-clocX, clocY)
            plocX, plocY = clocX, clocY

        # ========================
        # Click Mode
        # ========================
        if mode == "Click":
            length, img, lineInfo = detector.findDistance(8, 12, img)
            if length < 30:
                cv2.circle(img,(lineInfo[4], lineInfo[5]),
                           15, (0,255,0), cv2.FILLED)
                autopy.mouse.click()

        # ========================
        # Right Click
        # ========================
        if mode == "Right Click":
            autopy.mouse.click(button=autopy.mouse.Button.RIGHT)
            time.sleep(0.3)

        # ========================
        # Scroll Mode
        # ========================
        if mode == "Scroll":
            if y1 < hCam//2:
                autopy.mouse.scroll(10)
            else:
                autopy.mouse.scroll(-10)

        cv2.circle(img,(x1,y1),15,color, cv2.FILLED)
        cv2.circle(img,(x1,y1),25,(255,255,255),2)

 
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20,50),
                cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)

    cv2.putText(img, f'Mode: {mode}', (20,100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("Shatha AI Mouse", img)
    cv2.waitKey(1)
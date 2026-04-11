import cv2
import numpy as np 
import gesture_engine as htm
import time 
import autopy 

# Settings
wCam, hCam = 640, 480
frameR = 100

plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.GestureEngine(maxHands=1)
wScr, hScr = autopy.screen.size()

pTime = 0

# إنشاء النافذة مرة وحدة فقط
cv2.namedWindow("Shatha Gesture Engine", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Shatha Gesture Engine", 900, 700)
cv2.moveWindow("Shatha Gesture Engine", 300, 100)

# Main Loop
while True:
    success, img = cap.read()

    if not success:
        print("Camera error")
        break

    hands, img = detector.findHands(img)

    mode = "None"
    color = (255,0,255)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]

        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp(hand)

        # Modes
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

        # إطار
        cv2.rectangle(img,(frameR, frameR),(wCam-frameR, hCam-frameR), color, 2)

        # Move
        if mode == "Move":
            x3 = np.interp(x1, (frameR, wCam-frameR), (0,wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0,hScr))

            distance = np.hypot(x1 - plocX, y1 - plocY)
            smoothening = max(3, min(10, int(distance / 10)))

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            autopy.mouse.move(wScr-clocX, clocY)
            plocX, plocY = clocX, clocY

        # Click
        if mode == "Click":
            distData = detector.findDistance(8, 12, hand, img)
            if distData["length"] < 30:
                cx, cy = distData["center"]
                cv2.circle(img,(cx, cy),15,(0,255,0),cv2.FILLED)
                autopy.mouse.click()

        # Right Click
        if mode == "Right Click":
            autopy.mouse.click(button=autopy.mouse.Button.RIGHT)
            time.sleep(0.3)

        # Scroll
        if mode == "Scroll":
            if y1 < hCam//2:
                autopy.mouse.scroll(10)
            else:
                autopy.mouse.scroll(-10)

        # مؤشر
        cv2.circle(img,(x1,y1),15,color, cv2.FILLED)
        cv2.circle(img,(x1,y1),25,(255,255,255),2)

    # FPS
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20,50),
                cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)

    cv2.putText(img, f'Mode: {mode}', (20,100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    # عرض
    cv2.imshow("Shatha Gesture Engine", img)

    # خروج
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

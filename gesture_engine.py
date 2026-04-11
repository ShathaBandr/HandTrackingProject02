import cv2
import mediapipe as mp
import time
import math

class GestureEngine():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, trackCon=0.7):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )

        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    # Detect Hands
    def findHands(self, img, draw=True):
        if img is None:
            return [], img

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        allHands = []

        imgRGB.flags.writeable = False
        self.results = self.hands.process(imgRGB)
        imgRGB.flags.writeable = True
        if self.results.multi_hand_landmarks:
            for handNo, handLms in enumerate(self.results.multi_hand_landmarks):

                lmList = []
                xList, yList = [], []

                for id, lm in enumerate(handLms.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    lmList.append([id, cx, cy])
                    xList.append(cx)
                    yList.append(cy)

                bbox = (min(xList), min(yList), max(xList), max(yList))

                # حماية من crash
                if self.results.multi_handedness:
                    handType = self.results.multi_handedness[handNo].classification[0].label
                else:
                    handType = "Right"

                allHands.append({
                    "lmList": lmList,
                    "bbox": bbox,
                    "type": handType
                })

                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return allHands, img

    # Fingers Up
    def fingersUp(self, hand):
        fingers = []
        lmList = hand["lmList"]
        handType = hand["type"]

        # Thumb
        if handType == "Right":
            fingers.append(1 if lmList[self.tipIds[0]][1] > lmList[self.tipIds[0]-1][1] else 0)
        else:
            fingers.append(1 if lmList[self.tipIds[0]][1] < lmList[self.tipIds[0]-1][1] else 0)

        # باقي الأصابع
        for id in range(1, 5):
            fingers.append(1 if lmList[self.tipIds[id]][2] < lmList[self.tipIds[id]-2][2] else 0)

        return fingers

    # Gesture Detection
    def detectGesture(self, hand):
        fingers = self.fingersUp(hand)

        if fingers == [0,1,0,0,0]:
            return "Move"
        elif fingers == [0,1,1,0,0]:
            return "Click"
        elif fingers == [1,1,1,1,1]:
            return "Open Hand"
        elif fingers == [0,0,0,0,0]:
            return "Fist"
        elif fingers == [1,0,0,0,0]:
            return "Thumb"
        else:
            return "Unknown"

    # Distance
    def findDistance(self, p1, p2, hand, img, draw=True):
        lmList = hand["lmList"]

        x1, y1 = lmList[p1][1:]
        x2, y2 = lmList[p2][1:]
        cx, cy = (x1 + x2)//2, (y1 + y2)//2

        length = math.hypot(x2 - x1, y2 - y1)

        if draw:
            cv2.line(img, (x1,y1), (x2,y2), (255,0,255), 2)
            cv2.circle(img, (x1,y1), 8, (0,255,0), cv2.FILLED)
            cv2.circle(img, (x2,y2), 8, (0,255,0), cv2.FILLED)
            cv2.circle(img, (cx,cy), 8, (0,0,255), cv2.FILLED)

        return {
            "length": length,
            "center": (cx, cy)
        }


# Demo Main
def main():
    cap = cv2.VideoCapture(0)
    detector = GestureEngine()

    pTime = 0

    while True:
        success, img = cap.read()

        if not success:
            print("Camera failed")
            continue

        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            lmList = hand["lmList"]

            gesture = detector.detectGesture(hand)

            if len(lmList) > 8:
                x, y = lmList[8][1:]

                cv2.putText(img, f"Gesture: {gesture}", (10,70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

                cv2.circle(img, (x,y), 15, (255,0,255), cv2.FILLED)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, f"FPS: {int(fps)}", (10,120),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)

        cv2.imshow("Shatha Gesture Engine", img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

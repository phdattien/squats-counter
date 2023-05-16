import cv2
import mediapipe as mp
import time
import math
import numpy as np


class poseDetector():
    def __init__(self):
        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose()

    def find_pose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,
                                           self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def getPositions(self, img, draw=True):
        self.lmList = []

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(w*lm.x), int(h*lm.y)
                self.lmList.append([id, lm.z, cx, cy])
                if (draw):
                    cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)

        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.lmList[p1][2:]
        x2, y2 = self.lmList[p2][2:]
        x3, y3 = self.lmList[p3][2:]

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2)
                             - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        if angle > 180:
            angle = np.abs(angle - 360)

        if draw:
            # can be made prettier do it later when have time
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), 2)

            cv2.circle(img, (x2, y2), 10, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 + 20, y2),
                        cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 0), 3)
        return angle


def main():
    cTime = 0
    pTime = 0
    cap = cv2.VideoCapture("squat_male_small.mp4")
    if not cap.isOpened():
        print("Error opening video stream or file")

    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = detector.find_pose(img)

        cTime = time.time()
        fps = 1/(cTime - pTime)  # how many we can fit in one second
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow('Image', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

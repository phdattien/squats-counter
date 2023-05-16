import cv2
import mediapipe as mp
import time


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
        lmList = []

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(w*lm.x), int(h*lm.y)
                lmList.append([id, cx, cy])
                if (draw):
                    cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)

        return lmList


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

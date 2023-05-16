#!/usr/bin/env python3

import numpy as np
import cv2
import PoseModule as pm
import argparse

count = 0
dir = 0  # 0 means going down
detector = pm.poseDetector()

parser = argparse.ArgumentParser()
parser.add_argument("video_stream", nargs="?", default="0", type=str, help="Path to video")


def down_scale(img, scale):
    if img is None:
        return

    scale_percent = scale  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return img


def addSquat(img, lmList):
    """determines if squat is made by if knees make 90 degree"""
    global count
    global dir

    if len(lmList) == 0:
        return
    # right leg
    rAngle = detector.findAngle(img, 24, 26, 28)
    # right left
    lAngle = detector.findAngle(img, 23, 25, 27)

    per1 = np.interp(rAngle, (90, 175), (100, 0))
    per2 = np.interp(lAngle, (90, 175), (100, 0))

    if ((per1 == 100) or (per2 == 100)) and (dir == 0):
        count += 0.5
        dir = 1

    if ((per1 == 0) or (per2 == 0)) and (dir == 1):
        count += 0.5
        dir = 0


def main(args: argparse.Namespace):
    if args.video_stream == str(0):
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.video_stream)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    while True:
        success, img = cap.read()
        if not success:
            break

        # img = down_scale(img, 50)
        img = down_scale(img, 50)
        img = detector.find_pose(img)
        lmList = detector.getPositions(img, False)

        addSquat(img, lmList)

        cv2.putText(img, str(int(count)),
                    (int(img.shape[1]/2), 70),
                    cv2.FONT_HERSHEY_PLAIN, 5,
                    (255, 0, 0), 3)

        cv2.imshow('squat', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

# importing the necessary packages
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2 #computer vision
from imutils.video import VideoStream
from threading import Thread
import playsound
import argparse
import time


def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)

def eyeSound(path):
    while (ear < EYE_AR_THRESH and COUNTER >= EYE_AR_CONSEC_FRAMES):
    # play an alarm sound
	    playsound.playsound(path)
#calculating eye aspect ratio
def eye_aspect_ratio(eye):
	# compute the euclidean distances between the vertical
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	return ear

#calculating mouth aspect ratio
def mouth_aspect_ratio(mou):
	# compute the euclidean distances between the horizontal
	X   = dist.euclidean(mou[0], mou[6])
	# compute the euclidean distances between the vertical
	Y1  = dist.euclidean(mou[2], mou[10])
	Y2  = dist.euclidean(mou[4], mou[8])
	# taking average
	Y   = (Y1+Y2)/2.0
	# compute mouth aspect ratio
	mar = Y/X
	return mar

camera = cv2.VideoCapture(0)


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="",
	help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())
 
# define constants for aspect ratios
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48
MOU_AR_THRESH = 0.6

COUNTER = 0
COUNTER_2=0
ALARM_ON = False
ALARM_ON2 = False
framecount=0
yawnStatus = False
yawns = 0
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
#extract ace rom image
detector = dlib.get_frontal_face_detector()
#predict
predictor = dlib.shape_predictor(args["shape_predictor"]) #take wighets and start to predict

# grab the indexes of the facial landmarks for the left and right eye
# also for the mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# loop over captuing video
while True:
	# grab the frame from the camera, resize
	# it, and convert it to grayscale
	# channels)
    ret, frame = camera.read()
    framecount=framecount+1
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Obtain image_landmarks lip_distance from mouth_open function for current frame.
    prev_yawn_status = yawnStatus
	# detect faces in the grayscale frame
    rects = detector(gray, 0)
    # print(len(rects))
    if len(rects) == 0:
        COUNTER_2 += 1

        if COUNTER_2 >= 10:
            # if the alarm is not on, turn it on
            if not ALARM_ON:
                ALARM_ON = True

                # check to see if an alarm file was supplied,
                # and if so, start a thread to have the alarm
                # sound played in the background
                if args["alarm"] != "":
                    v = Thread(target=sound_alarm,
                        args=(args["alarm"],))
                    v.deamon = True
                    v.start()
            # draw an alarm on the frame
            cv2.putText(frame, "Driver Distracted!", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
	# loop over the face detections
    else:
        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            mouEAR = mouth_aspect_ratio(mouth)
            
            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 255), 1)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH :
                COUNTER += 1
                # if the eyes were closed for a sufficient number of
                # then sound the alarm
                if COUNTER >= EYE_AR_CONSEC_FRAMES  :
                    # if the alarm is not on, turn it on
                    if not ALARM_ON:
                        ALARM_ON = True

                        # check to see if an alarm file was supplied,
                        # and if so, start a thread to have the alarm
                        # sound played in the background
                        if args["alarm"] != "":
                            t = Thread(target=eyeSound,
                                args=(args["alarm"],))
                            t.deamon = True
                            t.start()

                    # draw an alarm on the frame
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # otherwise, the eye aspect ratio is not below the blink
            # threshold, so reset the counter and alarm
            else:
                COUNTER = 0
                ALARM_ON = False
                cv2.putText(frame, "Eyes Open ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, "EAR: {:.2f}".format(ear), (480, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # yawning detections

            if mouEAR > MOU_AR_THRESH:
                cv2.putText(frame, "Yawning ", (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                yawnStatus = True
                if (yawns == 3 and framecount >= 1000):
                    framecount=0
                    yawns=0
                    if not ALARM_ON2:
                        ALARM_ON2 = True
                        if args["alarm"] != "":
                            s = Thread(target=sound_alarm,
                                        args=(args["alarm"],))
                        s.deamon = True
                        s.start()
                output_text = "Yawn Count: " + str(yawns + 1)
                cv2.putText(frame, output_text, (10,100),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0),2)
            else:
                yawnStatus = False
                ALARM_ON2 = False

            if prev_yawn_status == True and yawnStatus == False:
                yawns+=1

            # cv2.putText(frame, "MAR: {:.2f}".format(mouEAR), (480, 60),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # cv2.putText(frame,"Lusip Project @ Swarnim",(370,470),cv2.FONT_HERSHEY_COMPLEX,0.6,(153,51,102),1)
    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
camera.release()

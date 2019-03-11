import face_recognition
from os import path
import cv2
import numpy
from imutils.video import VideoStream
import imutils
import time
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils import face_utils
import argparse
import dlib

class Face:
    def __init__(self, app):
        self.storage = app.config["storage"]
        self.db = app.db
        self.faces = []  # storage all faces in caches array of face object
        self.known_encoding_faces = []  # faces data for recognition
        self.face_user_keys = {}
        self.load_all()
        self.threshold = 0.5  # lower the value stricter the comparison of faces and better accuracy
        self.blink_cut_off_time = 3  # 5 seconds to blink

    def load_user_by_index_key(self, index_key=0):
        key_str = str(index_key)
        if key_str in self.face_user_keys:
            return self.face_user_keys[key_str]
        return None

    def load_train_file_by_name(self, name):
        trained_storage = path.join(self.storage, 'trained')
        return path.join(trained_storage, name)

    def load_unknown_file_by_name(self, name):
        unknown_storage = path.join(self.storage, 'unknown')
        return path.join(unknown_storage, name)

    def eye_aspect_ratio(self, eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])

        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        # return the eye aspect ratio
        return ear

    def get_user_by_id(self, user_id):
        user = {}
        results = self.db.select(
            'SELECT users.id, users.user_name as name, faces.id, faces.user_id, faces.filename,faces.created FROM vtiger_users as users LEFT JOIN covalense_faces as faces ON faces.user_id = users.id WHERE users.id = %s',
            [user_id])
        index = 0
        for row in results:
            face = {
                "id": row[2],
                "user_id": row[3],
                "filename": row[4],
                "created": row[5],
            }
            if index == 0:
                user = {
                    "id": row[0],
                    "name": row[1],
                    "faces": [],
                }
            if row[3]:
                user["faces"].append(face)
            index = index + 1

        if 'id' in user:
            return user
        return None

    def load_all(self):
        results = self.db.select('SELECT id, user_id, filename, encoding, created FROM covalense_faces')
        if results is not None:
            for row in results:
                user_id = row[1]
                filename = row[2]
                # encoding = json.loads(row[3])
                encoding = numpy.loads(row[3])
                face = {
                    "id": row[0],
                    "user_id": user_id,
                    "filename": filename,
                    "created": row[4]
                }
                self.faces.append(face)
                index_key = len(self.known_encoding_faces)
                self.known_encoding_faces.append(encoding)
                index_key_string = str(index_key)
                self.face_user_keys['{0}'.format(index_key_string)] = user_id

    def recognize(self, unknown_filename):
        unknown_image = face_recognition.load_image_file(self.load_unknown_file_by_name(unknown_filename))
        unknown_encoding_image = face_recognition.face_encodings(unknown_image)[0]

        # load the input image and convert it from BGR to RGB
        image = cv2.imread(self.load_unknown_file_by_name(unknown_filename))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes corresponding
        # to each face in the input image, then compute the facial embeddings
        # for each face
        print("[INFO] recognizing faces...")
        boxes = face_recognition.face_locations(rgb, model="cnn")
        encodings = face_recognition.face_encodings(rgb, boxes)

        # initialize the list of names for each face detected
        user_ids = []

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(self.known_encoding_faces, encoding, self.threshold)
            # print("matches")
            # print(matches)
            distances = face_recognition.face_distance(self.known_encoding_faces, encoding)
            # print("distances")
            # print(distances)
            min_dist = numpy.min(distances)
            index_of_minimum_dist = numpy.where(distances == min_dist)
            index_of_minimum_dist = index_of_minimum_dist[0][0]

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    # print("i")
                    # print(i)
                    # user_id = self.load_user_by_index_key(i)
                    # print("user_id")
                    # print(user_id)
                    if index_of_minimum_dist == i:
                        # so we found this user with index key and find him
                        user_id = self.load_user_by_index_key(i)
                        return user_id
        return None

    def blink_detection(self):
        # define two constants, one for the eye aspect ratio to indicate
        # blink and then a second constant for the number of consecutive
        # frames the eye must be below the threshold
        EYE_AR_THRESH = 0.25
        EYE_AR_CONSEC_FRAMES = 3
        # initialize the frame counters and the total number of blinks
        COUNTER = 0
        TOTAL = 0
        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        print("[INFO] loading facial landmark predictor...")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        # grab the indexes of the facial landmarks for the left and
        # right eye, respectively
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        # start the video stream thread
        print("[INFO] starting video stream thread...")
        # vs = FileVideoStream(args["video"]).start()
        # fileStream = True
        vs = VideoStream(src=0).start()
        # vs = VideoStream(usePiCamera=True).start()
        fileStream = False
        time.sleep(1.0)
        # loop over frames from the video stream
        start_blink_detection_time = time.time()
        while True:
            # if this is a file video stream, then we need to check if
            # there any more frames left in the buffer to process
            if fileStream and not vs.more():
                break

            # grab the frame from the threaded video file stream, resize
            # it, and convert it to grayscale
            # channels)
            frame = vs.read()
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale frame
            rects = detector(gray, 0)

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
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)

                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0

                # compute the convex hull for the left and right eye, then
                # visualize each of the eyes
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                # check to see if the eye aspect ratio is below the blink
                # threshold, and if so, increment the blink frame counter
                if ear < EYE_AR_THRESH:
                    COUNTER += 1

                # otherwise, the eye aspect ratio is not below the blink
                # threshold
                else:
                    # if the eyes were closed for a sufficient number of
                    # then increment the total number of blinks
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        TOTAL += 1
                        # do a bit of cleanup
                        cv2.destroyAllWindows()
                        print("blinked count - " + str(TOTAL))
                        time.sleep(2.0)
                        # proceed to face detection
                        return self.recognize_faces_in_video(vs)
                    else:
                        end_blink_detection_time = time.time()
                        if (end_blink_detection_time - start_blink_detection_time) > self.blink_cut_off_time:
                            # do a bit of cleanup
                            cv2.destroyAllWindows()
                            vs.stop()
                            return None
                    # reset the eye frame counter
                    COUNTER = 0

    def recognize_faces_in_video(self, vs):
        # initialize the video stream and pointer to output video file, then
        # allow the camera sensor to warm up
        print("[INFO] starting video stream...")
        # vs = VideoStream(src=0).start()
        writer = None
        time.sleep(2.0)
        print("[INFO] before while in recognize_faces_in_video...")
        # loop over frames from the video file stream
        while True:
            # grab the frame from the threaded video stream
            frame = vs.read()

            # convert the input frame from BGR to RGB then resize it to have
            # a width of 750px (to speedup processing)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = imutils.resize(frame, width=750)
            r = frame.shape[1] / float(rgb.shape[1])

            # detect the (x, y)-coordinates of the bounding boxes
            # corresponding to each face in the input frame, then compute
            # the facial embeddings for each face
            boxes = face_recognition.face_locations(rgb, model="cnn")
            encodings = face_recognition.face_encodings(rgb, boxes)
            names = []

            # initialize the list of names for each face detected
            user_ids = []
            print("[INFO] encoding detected...")
            # loop over the facial embeddings
            for encoding in encodings:
                # attempt to match each face in the input image to our known
                # encodings
                matches = face_recognition.compare_faces(self.known_encoding_faces, encoding, self.threshold)
                # print("matches")
                # print(matches)
                distances = face_recognition.face_distance(self.known_encoding_faces, encoding)
                # print("distances")
                # print(distances)
                min_dist = numpy.min(distances)
                index_of_minimum_dist = numpy.where(distances == min_dist)
                index_of_minimum_dist = index_of_minimum_dist[0][0]

                # check to see if we have found a match
                if True in matches:
                    # find the indexes of all matched faces then initialize a
                    # dictionary to count the total number of times each face
                    # was matched
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}

                    # loop over the matched indexes and maintain a count for
                    # each recognized face face
                    for i in matchedIdxs:
                        # print("i")
                        # print(i)
                        user_id = self.load_user_by_index_key(i)
                        print("user_id")
                        print(user_id)
                        if index_of_minimum_dist == i:
                            # so we found this user with index key and find him
                            user_id = self.load_user_by_index_key(i)
                            cv2.destroyAllWindows()
                            vs.stop()
                            return user_id
            cv2.destroyAllWindows()
            vs.stop()
            return None
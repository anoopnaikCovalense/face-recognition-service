import face_recognition
from os import path
import cv2
import numpy
from imutils.video import VideoStream
import imutils
import time

class Face:
    def __init__(self, app):
        self.storage = app.config["storage"]
        self.db = app.db
        self.faces = []  # storage all faces in caches array of face object
        self.known_encoding_faces = []  # faces data for recognition
        self.face_user_keys = {}
        self.load_all()
        self.threshold = 0.5

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

    def recognize_faces_in_video(self):
        # initialize the video stream and pointer to output video file, then
        # allow the camera sensor to warm up
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        writer = None
        time.sleep(2.0)

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
                            vs.stop()
                            return user_id
            vs.stop()
            return None
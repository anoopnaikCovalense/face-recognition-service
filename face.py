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
            'SELECT users.id, users.name, users.created, faces.id, faces.user_id, faces.filename,faces.created FROM users LEFT JOIN faces ON faces.user_id = users.id WHERE users.id = %s',
            [user_id])

        index = 0
        for row in results:
            face = {
                "id": row[3],
                "user_id": row[4],
                "filename": row[5],
                "created": row[6],
            }
            if index == 0:
                user = {
                    "id": row[0],
                    "name": row[1],
                    "created": row[2],
                    "faces": [],
                }
            if row[3]:
                user["faces"].append(face)
            index = index + 1

        if 'id' in user:
            return user
        return None

    def load_all(self):

        results = self.db.select('SELECT faces.id, faces.user_id, faces.filename, faces.encoding, faces.created FROM faces')
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
            matches = face_recognition.compare_faces(self.known_encoding_faces, encoding)
            distances = face_recognition.face_distance(self.known_encoding_faces, encoding)
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
                matches = face_recognition.compare_faces(self.known_encoding_faces, encoding)
                distances = face_recognition.face_distance(self.known_encoding_faces, encoding)
                print("distances")
                print(distances)
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
                        if index_of_minimum_dist == i:
                            # so we found this user with index key and find him
                            user_id = self.load_user_by_index_key(i)
                            vs.stop()
                            return user_id
            return None

            # loop over the recognized faces
            for ((top, right, bottom, left), user) in zip(boxes, user_ids):
                # rescale the face coordinates
                top = int(top * r)
                right = int(right * r)
                bottom = int(bottom * r)
                left = int(left * r)

                userResult = self.get_user_by_id(user_id)
                name = userResult["name"]

                # draw the predicted face name on the image
                cv2.rectangle(frame, (left, top), (right, bottom),
                              (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)

            # if the video writer is None *AND* we are supposed to write
            # the output video to disk initialize the writer
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter("/home/an/Documents/facerecognition/face-recognition-service/", fourcc, 20,
                                         (frame.shape[1], frame.shape[0]), True)

            # if the writer is not None, write the frame with recognized
            # faces to disk
            if writer is not None:
                writer.write(frame)

            # check to see if we are supposed to display the output frame to
            # the screen
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()

        # check to see if the video writer point needs to be released
        if writer is not None:
            writer.release()
        return None
from flask import Flask, json, Response, request, render_template
from werkzeug.utils import secure_filename
from os import path, getcwd
import time
from db import Database
from face import Face
import cv2
import face_recognition
import numpy
from base64 import b64decode
import sys

app = Flask(__name__)
app.config['file_allowed'] = ['image/png', 'image/jpeg']
app.config['storage'] = path.join(getcwd(), 'storage')
app.db = Database()
app.face = Face(app)

def success_handle(output, status=200, mimetype='application/json'):
    return Response(output, status=status, mimetype=mimetype)


def error_handle(error_message, status=500, mimetype='application/json'):
    return Response(json.dumps({"error": {"message": error_message}}), status=status, mimetype=mimetype)


def get_user_by_id(user_id):
    user = {}
    results = app.db.select(
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

def delete_user_by_id(user_id):
    app.db.delete('DELETE FROM users WHERE users.id = %s', [user_id])
    # also delete all faces with user id
    app.db.delete('DELETE FROM faces WHERE faces.user_id = %s', [user_id])

#   Route for Hompage
@app.route('/', methods=['GET'])
def page_home():

    return render_template('index.html')

@app.route('/api', methods=['GET'])
def homepage():
    output = json.dumps({"api": '1.0'})
    return success_handle(output)

@app.route('/api/train', methods=['POST'])
def train():
    output = json.dumps({"success": True})

    if ('file' not in request.files) and (request.form['image'] == ""):
            print("Face image is required")
            return error_handle("Face image is required.")
    else:
        print("File request", request.files)
        image_data_uri = request.form['image']

        if image_data_uri != "undefined":
            header, image_data_uri = image_data_uri.split(",", 1)
            name = request.form['name']
            image_data_uri = "+".join(image_data_uri.split(" "))
            binary_data = b64decode(image_data_uri)
            trained_storage = path.join(app.config['storage'], 'trained')
            saved_file_path = path.join(trained_storage, name)
            created_file_ts = str(time.time())
            filename = name + created_file_ts + '.jpg'
            saved_file_path = saved_file_path + created_file_ts + '.jpg'
            fd = open(saved_file_path, 'wb')
            fd.write(binary_data)
            fd.close()

        print(saved_file_path)

        # if image is captured using webcam
        if image_data_uri == "undefined":
            file = request.files['file']

        # if image type is not valid and image was not captured using webcam
        if image_data_uri == "undefined" and file.mimetype not in app.config['file_allowed']:
            print("File extension is not allowed")
            return error_handle("We are only allow upload file with *.png , *.jpg")
        else:
            # get name in form data
            name = request.form['name']
            
            if image_data_uri == "undefined":
                print("Information of that face", name)
                print("File is allowed and will be saved in ", app.config['storage'])
                filename = secure_filename(file.filename)
                trained_storage = path.join(app.config['storage'], 'trained')
                saved_file_path = path.join(trained_storage, filename)
                file.save(saved_file_path)

            if "id" in request.form and request.form['id']:
                user_id = request.form['id']

            # load the input image and convert it from RGB (OpenCV ordering)
            # to dlib ordering (RGB)
            image = cv2.imread(saved_file_path)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # detect the (x, y)-coordinates of the bounding boxes
            # corresponding to each face in the input image
            boxes = face_recognition.face_locations(rgb, model="cnn")
            # compute the facial embedding for the face
            encodings = face_recognition.face_encodings(rgb, boxes)
            # loop over the encodings
            for encoding in encodings:
                # add each encoding + name to our set of known names and
                # encodings in mysql database
                created = int(time.time())

                if 'user_id' not in locals():
                    print("Inserting user in db")
                    user_id = app.db.insert('INSERT INTO users(name, created) values(%s,%s)', [name, str(created)])

                if user_id:
                    print("User saved in data", name, user_id)
                    # user has been save with user_id and now we need save faces table as well

                    # encoding_pickle = pickle.dumps(encoding, protocol=0)
                    # encoding_json = json.dumps(encoding)

                    encoding_pickle = numpy.ndarray.dumps(encoding)

                    face_id = app.db.insert('INSERT INTO covalense_faces(user_id, filename, encoding, created) values(%s,%s,%s,%s)',
                                            [user_id, filename, encoding_pickle, str(created)])

                    # append this entry in the known encodings in face.py to prevent restart of service every time
                    # a new face is added to database because all the encodings of users is loaded in memory when service starts
                    index_key = len(app.face.known_encoding_faces)
                    app.face.known_encoding_faces.append(encoding)
                    index_key_string = str(index_key)
                    app.face.face_user_keys['{0}'.format(index_key_string)] = user_id

                    if face_id:
                        print("cool face has been saved")
                        face_data = {"id": face_id, "filename": filename, "created": created}
                        return_output = json.dumps({"id": user_id, "name": name, "face": [face_data]})
                        return success_handle(return_output)
                    else:
                        print("An error saving face image.")
                        return error_handle("An error saving face image.")
                else:
                    print("Something happened")
                    return error_handle("An error inserting new user")
        print("Request is contain image")
    return success_handle(output)


# route for user profile
@app.route('/api/users/<int:user_id>', methods=['GET', 'DELETE'])
def user_profile(user_id):
    if request.method == 'GET':
        user = get_user_by_id(user_id)
        if user:
            return success_handle(json.dumps(user), 200)
        else:
            return error_handle("User not found", 404)
    if request.method == 'DELETE':
        delete_user_by_id(user_id)
        return success_handle(json.dumps({"deleted": True}))


# router for recognize a unknown face
@app.route('/api/recognize', methods=['POST'])
def recognize():
    if 'file' not in request.files:
        return error_handle("Image is required")
    else:
        file = request.files['file']
        # file extension valid
        if file.mimetype not in app.config['file_allowed']:
            return error_handle("File extension is not allowed")
        else:
            filename = secure_filename(file.filename)
            unknown_storage = path.join(app.config["storage"], 'unknown')
            file_path = path.join(unknown_storage, filename)
            file.save(file_path)

            user_id = app.face.recognize(filename)
            if user_id is not None:
                user = get_user_by_id(user_id)
                message = {"message": "Hey we found {0} matched with your face image".format(user["name"]),
                           "user": user}
                return success_handle(json.dumps(message))
            else:
                return error_handle("Sorry we can not found any people matched with your face image, try another image")

# router for recognize a unknown face
@app.route('/api/recognizeFacesInVideo', methods=['GET'])
def recognize_faces_in_video():
    user_id = app.face.recognize_faces_in_video()
    if user_id is not None:
        user = get_user_by_id(user_id)
        message = {"message": "Hey we found {0} matched with your face image".format(user["name"]),
                   "user": user}
        return success_handle(json.dumps(message))
    else:
        return error_handle("Sorry we can not found any people matched with your face image, try another image")

# Run the app
app.run(host="0.0.0.0")

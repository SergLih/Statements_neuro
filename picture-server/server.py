import socket
import argparse
import os
from tqdm import tqdm
from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import json

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--port",     required=True, help="server port", default=9876)
ap.add_argument("-m", "--models",   help="path to folder with trained models", default='models')
ap.add_argument("-e", "--embedder", help="path to OpenCV's deep learning face embedding model",
									default='models/openface_nn4.small2.v1.t7')
ap.add_argument("-d", "--detector", help="path to OpenCV's deep learning face detector model",
									default='models/res10_300x300_ssd_iter_140000.caffemodel')
ap.add_argument("-r", "--detector-proto", help="path to OpenCV's deep learning face detector model's prototxt",
									default='models/res10_300x300_ssd_iter_140000.caffemodel.prototxt')
ap.add_argument("-f", "--files",    required=True, help="path to root of images file structure")
ap.add_argument("-u", "--unknown",  help="path to the folder with images of unknown people", default='unknown/')
args = vars(ap.parse_args())
args['files'] = os.path.expanduser(args['files'])
print(args['files'])

HOST = ''
PORT = int(args['port'])
ADDR = (HOST,PORT)
BUFSIZE = 4096
SIGNATURE = b'lihrazum9876543210!@#$%^&*()'
LS = len(SIGNATURE)
CONF = 0.5

print("[INFO] loading face detector...")
detector = cv2.dnn.readNetFromCaffe(args["detector_proto"], args["detector"])
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedder"])

serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serv.bind(ADDR)
serv.listen(5)
print('[INFO]\tServer is waiting for client\'s connection...')



def classify(group_id, photo_aud_path):
	photo_aud_path =  os.path.join(args["files"], photo_aud_path)
	print("[INFO]\tCLASSIFYING " + photo_aud_path)
	with open(os.path.join(args['models'], "group" + str(group_id) + "model.pkl"), "rb") as f:
		data = pickle.loads(f.read())

	le         = data['label_encoder']
	recognizer = data['recognizer']

	# load the image, resize it to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image dimensions
	image = cv2.imread(photo_aud_path)
	# image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		image, #cv2.resize(image, (300, 300)),
		1.0, (w, h), #(300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# loop over the detections
	list_faces = []
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > CONF:
			# compute the (x, y)-coordinates of the bounding box for the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI
			face = image[startY:endY, startX:endX]
			face = imutils.resize(face, width=96)
			#cv2.imshow(str(i), face)
			#cv2.waitKey(0)
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
				(0, 0, 0), swapRB=False, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]
			#print("[INFO] found:" + name )
			if name != "_unknown_":
				list_faces.append(name)
			# draw the bounding box of the face along with the associated probability
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(image, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			# cv2.putText(image, text, (startX, y),
				# cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

			cv2.putText(image, text, (startX, y),
				fontFace=cv2.FONT_HERSHEY_COMPLEX , fontScale=0.4, color=[0, 0, 0], thickness=2)
			cv2.putText(image, text, (startX, y),
				fontFace=cv2.FONT_HERSHEY_COMPLEX , fontScale=0.4, color=[255, 255, 255], thickness=1)


	def get_faces_path(photo_aud_path):
		path_parts = list(os.path.split(photo_aud_path))
		path_parts[-1] = 'faces_' + path_parts[-1]
		return os.path.join(*path_parts)

	new_path = get_faces_path(photo_aud_path)
	print(new_path)
	cv2.imwrite(new_path, image)
	print("[INFO] file with faces is ready...")
	list_faces = list(set(list_faces))
	print(list_faces)
	return list_faces



def train_model(group_id):
	group_data_path = os.path.join(args["files"], 'students', "group" + str(group_id))
	print("[INFO] RETRAINING MODEL AT " + group_data_path)

	print("[INFO] quantifying faces...")
	imagePaths = list(paths.list_images(group_data_path))
	unknownPaths = list(paths.list_images(args["unknown"]))
	# initialize our lists of extracted facial embeddings and corresponding people names
	knownEmbeddings, knownNames = [], []

	def process_image(imagePath, name):
		# load the image, resize it to have a width of 600 pixels (while
		# maintaining the aspect ratio), and then grab the image dimensions
		image = cv2.imread(imagePath)
		image = imutils.resize(image, width=600)
		(h, w) = image.shape[:2]

		# construct a blob from the image
		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(image, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)

		# apply OpenCV's deep learning-based face detector to localize faces in the input image
		detector.setInput(imageBlob)
		detections = detector.forward()

		# ensure at least one face was found
		if len(detections) > 0:
			# we're making the assumption that each image has only ONE
			# face, so find the bounding box with the largest probability
			i = np.argmax(detections[0, 0, :, 2])
			confidence = detections[0, 0, i, 2]

			# ensure that the detection with the largest probability also
			# means our minimum probability test (thus helping filter out weak detections)
			if confidence > CONF:
				# compute the (x, y)-coordinates of the bounding box for the face
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# extract the face ROI and grab the ROI dimensions
				face = image[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

				# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					return

				# construct a blob for the face ROI, then pass the blob
				# through our face embedding model to obtain the 128-d
				# quantification of the face
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					(96, 96), (0, 0, 0), swapRB=False, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()

				# add the name of the person + corresponding face embedding to their respective lists
				knownNames.append(name)
				knownEmbeddings.append(vec.flatten())

	print("[INFO] processing images for persons...")
	for (i, imagePath) in tqdm(enumerate(imagePaths)):
		name = imagePath.split(os.path.sep)[-2]
		process_image(imagePath, name)

	print("[INFO] processing images for unknown persons...")
	for (i, imagePath) in tqdm(enumerate(unknownPaths)):
		process_image(imagePath, "_unknown_")

	data = {"embeddings": knownEmbeddings, "names": knownNames}

	print("[INFO] encoding labels...")
	le = LabelEncoder()  #sklearn
	labels = le.fit_transform(data["names"])

	# train the model used to accept the 128-d embeddings of the face and
	# then produce the actual face recognition
	print("[INFO] training model...")
	recognizer = SVC(C=1.0, kernel="linear", probability=True)
	recognizer.fit(data["embeddings"], labels)

	# write the actual face recognition model to disk
	with open(os.path.join(args['models'], "group" + str(group_id) + "model.pkl"), "wb") as f:
		f.write(pickle.dumps({'recognizer': recognizer, 'label_encoder': le}))


def get_data(conn):
	while True:
		print('.', end='')
		data = conn.recv(BUFSIZE)
		print(len(data))
		if not data:
			conn.close()
			return data

while True:
	try:
		conn, addr = serv.accept()
		print('[INFO]\tClient connected, ', addr)
		data = conn.recv(BUFSIZE)
		if data[:LS] == SIGNATURE:
			if chr(data[LS]) == 'c':
				group_id = int.from_bytes(data[LS+1:LS+5], byteorder='big')
				photo_aud_path = data[LS+5:].decode('ascii')
				print('[INFO]\tClient requested to classify persons from group ', group_id)
				persons = classify(group_id, photo_aud_path)
				print('[INFO]\tSending response to the client...', end='')
				conn.send(bytearray(SIGNATURE) + bytearray(json.dumps(persons).encode('ascii')))
				print('ready')
				#with open('_2.jpg', 'wb') as f:
				#	f.write(data[LS+5:])
				# print('file is ready')
			elif chr(data[LS]) == 'u':
				group_id = int.from_bytes(data[LS+1:LS+5], byteorder='big')
				print('[INFO]\tClient requested to update the model for group ', group_id)
				train_model(group_id)
		else:
			print('[ERROR]\tInvalid request!')

		conn.close()
		print('[INFO]\tClient disconnected, ', addr)
	except KeyboardInterrupt:
		print('[INFO]\tServer has been stopped.')
		break

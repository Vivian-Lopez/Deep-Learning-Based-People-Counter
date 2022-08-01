# Importing the necessary libraries including flask and openCV
from flask import Flask, render_template, Response
import cv2
import cv2
import numpy as np
import argparse
import tensorflow as tf
import dlib

from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops

from trackable_object import TrackableObject
from centroidtracker import CentroidTracker

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1
# Patch the location of gfile
tf.gfile = tf.io.gfile

# Creating the flask app
app = Flask(__name__)

# Creating variable a camera which will store input from the webcam
camera = cv2.VideoCapture(0)

# Using a decorator and a function to render the html page located in the template folder
@app.route('/')
def index():
    return render_template('index.html')

def run_inference(model, category_index, cap, labels, roi_position=0.6, threshold=0.5, x_axis=True, skip_frames=20):
    counter = [0, 0, 0, 0]  # left, right, up, down
    total_frames = 0

    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    while cap.isOpened():
        ret, image_np = cap.read()
        if not ret:
            break

        height, width, _ = image_np.shape
        rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        status = "Waiting"
        rects = []

        if total_frames % skip_frames == 0:
            status = "Detecting"
            trackers = []

            # Actual detection.
            output_dict = run_inference_for_single_image(model, image_np)

            for i, (y_min, x_min, y_max, x_max) in enumerate(output_dict['detection_boxes']):
                if output_dict['detection_scores'][i] > threshold and (labels == None or category_index[output_dict['detection_classes'][i]]['name'] in labels):
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(
                        int(x_min * width), int(y_min * height), int(x_max * width), int(y_max * height))
                    tracker.start_track(rgb, rect)
                    trackers.append(tracker)
        else:
            status = "Tracking"
            for tracker in trackers:
                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                x_min, y_min, x_max, y_max = int(pos.left()), int(
                    pos.top()), int(pos.right()), int(pos.bottom())

                # add the bounding box coordinates to the rectangles list
                rects.append((x_min, y_min, x_max, y_max))

        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)

            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                if x_axis and not to.counted:
                    x = [c[0] for c in to.centroids]
                    direction = centroid[0] - np.mean(x)

                    if centroid[0] > roi_position*width and direction > 0 and np.mean(x) < roi_position*width:
                        counter[1] += 1
                        to.counted = True
                    elif centroid[0] < roi_position*width and direction < 0 and np.mean(x) > roi_position*width:
                        counter[0] += 1
                        to.counted = True

                elif not x_axis and not to.counted:
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)

                    if centroid[1] > roi_position*height and direction > 0 and np.mean(y) < roi_position*height:
                        counter[3] += 1
                        to.counted = True
                    elif centroid[1] < roi_position*height and direction < 0 and np.mean(y) > roi_position*height:
                        counter[2] += 1
                        to.counted = True

                to.centroids.append(centroid)

            trackableObjects[objectID] = to

            text = "ID {}".format(objectID)
            cv2.putText(image_np, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(
                image_np, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

        # Draw ROI line
        if x_axis:
            cv2.line(image_np, (int(roi_position*width), 0),
                     (int(roi_position*width), height), (0xFF, 0, 0), 5)
        else:
            cv2.line(image_np, (0, int(roi_position*height)),
                     (width, int(roi_position*height)), (0xFF, 0, 0), 5)

        # display count and status
        font = cv2.FONT_HERSHEY_SIMPLEX
        if x_axis:
            cv2.putText(image_np, f'Left: {counter[0]}; Right: {counter[1]}', (
                10, 35), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
        else:
            cv2.putText(image_np, f'Up: {counter[2]}; Down: {counter[3]}', (
                10, 35), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(image_np, 'Status: ' + status, (10, 70), font,
                    0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)

        total_frames += 1
def load_model(model_path):
    tf.keras.backend.clear_session()
    model = tf.saved_model.load(model_path)
    return model
    
def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(
        np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

# A function which takes the global camera variable and uses it to return a modified frame with a bounding box around faces face
def generate_frames(model, category_index, cap, labels, roi_position=0.6, threshold=0.5, x_axis=True, skip_frames=20,total_frames=0):
    while True:
            
        # read a frame from the camera
        success,image_np=camera.read()

        # Check is made whether the camera input was successfully read or not
        if not success:
            break
        else:

            ####################################################################

            height, width, _ = image_np.shape
            rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            status = "Waiting"
            rects = []

            if total_frames % skip_frames == 0:
                status = "Detecting"
                trackers = []

                # Actual detection.
                output_dict = run_inference_for_single_image(model, image_np)
                for i, (y_min, x_min, y_max, x_max) in enumerate(output_dict['detection_boxes']):
                    if output_dict['detection_scores'][i] > threshold: #and (labels == None or category_index[output_dict['detection_classes'][i]]['name'] in labels):
                        # print("DETECTION MADE")
                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(
                            int(x_min * width), int(y_min * height), int(x_max * width), int(y_max * height))
                        tracker.start_track(rgb, rect)
                        trackers.append(tracker)
            else:
                status = "Tracking"
                for tracker in trackers:
                    # update the tracker and grab the updated position
                    tracker.update(rgb)
                    pos = tracker.get_position()

                    # unpack the position object
                    x_min, y_min, x_max, y_max = int(pos.left()), int(
                        pos.top()), int(pos.right()), int(pos.bottom())

                    # add the bounding box coordinates to the rectangles list
                    rects.append((x_min, y_min, x_max, y_max))

            objects = ct.update(rects)

            for (objectID, centroid) in objects.items():
                to = trackableObjects.get(objectID, None)

                if to is None:
                    to = TrackableObject(objectID, centroid)
                else:
                    if x_axis and not to.counted:
                        x = [c[0] for c in to.centroids]
                        direction = centroid[0] - np.mean(x)

                        if centroid[0] > roi_position*width and direction > 0 and np.mean(x) < roi_position*width:
                            counter[1] += 1
                            to.counted = True
                        elif centroid[0] < roi_position*width and direction < 0 and np.mean(x) > roi_position*width:
                            counter[0] += 1
                            to.counted = True

                    elif not x_axis and not to.counted:
                        y = [c[1] for c in to.centroids]
                        direction = centroid[1] - np.mean(y)

                        if centroid[1] > roi_position*height and direction > 0 and np.mean(y) < roi_position*height:
                            counter[3] += 1
                            to.counted = True
                        elif centroid[1] < roi_position*height and direction < 0 and np.mean(y) > roi_position*height:
                            counter[2] += 1
                            to.counted = True

                    to.centroids.append(centroid)

                trackableObjects[objectID] = to

                text = "ID {}".format(objectID)
                cv2.putText(image_np, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.circle(
                    image_np, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

            # Draw ROI line
            if x_axis:
                cv2.line(image_np, (int(roi_position*width), 0),
                        (int(roi_position*width), height), (0xFF, 0, 0), 5)
            else:
                cv2.line(image_np, (0, int(roi_position*height)),
                        (width, int(roi_position*height)), (0xFF, 0, 0), 5)

            # display count and status
            font = cv2.FONT_HERSHEY_SIMPLEX
            net_left_count = counter[0] - counter[1]
            net_right_count = counter[1] - counter[0]
            if x_axis:
                cv2.putText(image_np, f'Left: {counter[0]}  Right: {counter[1]}', (10, 35), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
            else:
                cv2.putText(image_np, f'Up: {counter[2]}  Down: {counter[3]}', (10, 35), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
            if net_left_count >= 0:
                cv2.putText(image_np, 'Total Count: ' + str(net_left_count), (10, 70), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
            else:
                cv2.putText(image_np, 'Total Count: ' + str(net_right_count), (10, 70), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
            #cv2.putText(image_np, 'Status: ' + status, (10, 70), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)

            total_frames += 1
            ####################################################################
    
            # The resulting frame is encoded and converted to bytes
            ret,buffer=cv2.imencode('.jpg',image_np)
            frame=buffer.tobytes()

        # Yield is used so that the generator function can be continually called
        yield(b'--frame\r\n'b'Content-Type: frame/jpeg\r\n\r\n' + frame + b'\r\n')

# Here a decorator is used to start the generator function to allow for the video to be taken and processed and then returned to the webpage
@app.route('/video')
def video():
    return Response(generate_frames(detection_model, category_index, camera, labels="label_map.pbtxt"),mimetype='multipart/x-mixed-replace; boundary=frame')

# Creating a function to save the net counts
@app.route('/save_result')
def save_result(net_left_count, net_right_count):
    # Opens file and writes required text
    with open("Saved_counts.txt", 'w') as f:
        f.write("Net left count: ")
        f.write(net_left_count)
        f.write("\n")
        f.write("Net right count: ")
        f.write(net_right_count)

# The flask app itself is run in the following section with the host ip address and port specified
if __name__ == '__main__':
    detection_model = load_model("my_ssd_mobnet_tuned_2\export\saved_model")
    category_index = label_map_util.create_category_index_from_labelmap(
        "label_map.pbtxt", use_display_name=True)
    counter = [0, 0, 0, 0]  # left, right, up, down
    total_frames = 0

    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}
    app.run(host='127.0.0.1', debug=False, port=9999)
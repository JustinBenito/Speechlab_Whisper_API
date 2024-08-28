import numpy as np
import tensorflow as tf
import cv2

import os
print(os.getcwd())
# Load the pre-trained MobileNet-SSD model
model = tf.saved_model.load("model")

# Load the label map (for human detection, index 1 typically corresponds to a 'person')
category_index = {1: {'id': 1, 'name': 'person'}}

def preprocess_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB
    input_tensor = tf.convert_to_tensor(image_rgb)  # Convert to Tensor
    input_tensor = input_tensor[tf.newaxis,...]  # Add batch dimension
    return input_tensor

def detect_passengers(image):
    input_tensor = preprocess_image(image)
    
    # Perform object detection
    detections = model(input_tensor)
    
    # Extract relevant data
    num_detections = int(detections.pop('num_detections'))
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_scores = detections['detection_scores'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int64)
    
    return detection_boxes, detection_scores, detection_classes, num_detections

def draw_boxes_and_count_passengers(image, detection_boxes, detection_scores, detection_classes, score_threshold=0.5):
    passenger_count = 0
    height, width, _ = image.shape

    for i in range(len(detection_scores)):
        if detection_scores[i] > score_threshold and detection_classes[i] == 1:  # Only count people (class 1)
            passenger_count += 1
            
            # Get bounding box coordinates
            ymin, xmin, ymax, xmax = detection_boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
            
            # Draw bounding box on the image
            cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
    
    return passenger_count, image

def detect_and_count_passengers(image_path):
    # Load image
    image = cv2.imread(image_path)
    
    # Detect passengers
    detection_boxes, detection_scores, detection_classes, _ = detect_passengers(image)
    
    # Draw bounding boxes and count passengers
    passenger_count, image_with_boxes = draw_boxes_and_count_passengers(image, detection_boxes, detection_scores, detection_classes)
    
    print(f"Number of passengers detected: {passenger_count}")
    
    # Display the result
    cv2.imshow('Passengers Detected', image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return passenger_count, image_with_boxes

def save_output_image(image, output_path="output.jpg"):
    cv2.imwrite(output_path, image)

image_path = 'try.jpg'  # Path to the image file
passenger_count, output_image = detect_and_count_passengers(image_path)

# Optionally save the output image with bounding boxes
save_output_image(output_image, 'passengers_detected.jpg')

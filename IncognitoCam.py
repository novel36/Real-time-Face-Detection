# Import the necessary libraries
import cv2
from cvzone.FaceDetectionModule import FaceDetector


# Create a FaceDetector object with a minimum detection confidence of 0.75
detector = FaceDetector(minDetectionCon=0.75)

# Create a VideoCapture object to capture video from the default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Set the frame width and height to 640 and 480, respectively
cap.set(3, 1280)
cap.set(4, 720)

# Start an infinite loop
while True:
    # Read a frame from the video capture object
    success, img = cap.read()

    # Detect faces in the frame and draw their bounding boxes
    img, bboxs = detector.findFaces(img, draw=True)

    # Check if any faces were detected
    if bboxs:
        # Loop over the detected faces
        for i, bbox in enumerate(bboxs):
            # Get the bounding box coordinates of the current face
            x, y, w, h = bbox['bbox']

            # Crop the region of the frame corresponding to the current face
            imgCrop = img[y:y+h, x:x+w]


            # Check if the cropped image is not empty
            if imgCrop.size > 0:
                # Apply a blur effect to the cropped image
                
                imgBlur = cv2.blur(imgCrop, (35, 35))

                # Copy the blurred image back onto the original frame at its corresponding location
                img[y: y + h, x: x + w] = imgBlur

    # Display the processed frame in a window named "image"
    cv2.imshow("image", img)

    # Wait for 1 millisecond to update the window
    cv2.waitKey(1)
#Detector.py outputs the output.png based on the prediction made from the trained model
#note : multiple detection can be made by implementing face detection.

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
import matplotlib.pyplot as plt   

opencv_dnn_model = cv2.dnn.readNetFromCaffe(prototxt="face_detector/deploy.prototxt",caffeModel="face_detector/res10_300x300_ssd_iter_140000.caffemodel")

def face_mask_prediction(img):

    ap = argparse.ArgumentParser()

    model = load_model("output//maskdetector.model")
    hmodel = load_model("output//helmetdetector.model")

    ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())
    # image = cv2.imread(args["image"])
    image = img.copy()
    orig = image.copy()

    orig2 = image.copy()
    orig = cv2.resize(orig, (600, 600))

    min_confidence=0.5
    image_height, image_width, _ = orig2.shape

    output_image = orig2.copy()

    preprocessed_image = cv2.dnn.blobFromImage(orig2, scalefactor=1.0, size=(300, 300),
                                                mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)

    opencv_dnn_model.setInput(preprocessed_image)
        
    results = opencv_dnn_model.forward()    

    for face in results[0][0]:
            
        face_confidence = face[2]    
        if face_confidence > min_confidence:

            bbox = face[3:]
            x1 = int(bbox[0] * image_width)
            y1 = int(bbox[1] * image_height)
            x2 = int(bbox[2] * image_width)
            y2 = int(bbox[3] * image_height)

            image = cv2.resize(orig, (224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)
            image = np.expand_dims(image, axis=0)
            
            # pass the face through the model to determine if the face     :: has a mask or not

            (mask, withoutMask) = model.predict(image)[0]
            (helmet, withoutHelmet) = hmodel.predict(image)[0]

            label = "Mask" if mask > withoutMask else "No Mask"
            # color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            hlabel = "Helmet" if helmet > withoutHelmet else "No Helmet"
            # color = (255, 0, 0) if hlabel == "Helmet" else (0, 0, 255)

            print("label",label,"hlabel",hlabel)

            # Set Color for rectangle and label
            if label == "Mask" and hlabel == "Helmet":
                color = ( 0 , 255, 0 )
            elif label == "No Mask" and hlabel == "No Helmet":
                color = ( 0, 0 ,255)
            else:
                print("blue")
                color = ( 255 ,0 ,0)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            hlabel = "{}: {:.2f}%".format(hlabel, max(helmet, withoutHelmet) * 100)

            # print("label",label,"hlabel",hlabel)

            # Draw rectangle around the faces  

            cv2.rectangle(output_image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=image_width//200)

            # display the label and bounding box rectangle on the output 

            cv2.putText(output_image, text=label, org=(x1, y1-15), 
                            fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5,
                            color=color, thickness=1),
            cv2.putText(output_image, text=hlabel, org=(x1, y1-35), 
                            fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5,
                            color=color, thickness=1),

            output_image = cv2.resize(output_image, (600, 600))
      
    return output_image

# # Image recognition code

img = cv2.imread('testImages//testImage4.jpg')
image = face_mask_prediction(img)

cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.imwrite("output.jpg",image)


#Real Time Face Mask and Helmet Detection

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frames = cap.read()
#     if ret == False:
#         break
        
#     image = face_mask_prediction(frames)
#     cv2.imshow('Face Mask & Helmet Prediction',image)
#     if cv2.waitKey(1) == 27:
#         break
        
# cap.release()
# cv2.destroyAllWindows()
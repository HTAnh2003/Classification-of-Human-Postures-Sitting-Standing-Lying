# %%
import psycopg2
import pytube
import os
import cv2
import matplotlib.pyplot as plt
import cv2
import numpy as np
import joblib
import cv2 
import json
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import pickle

# %%

# Load weight model
# Tải lại mô hình SVM đã lưu
pickle_file = r"D:\HocTap\HK2_Nam3\ML\Final-machine-learning\svm_model_sit_stand_lie.pkl"
with open(pickle_file, 'rb') as file:
    loaded_svm_model = pickle.load(file)

# Load weight model stand and lie
# with open(r'D:\HocTap\HK2_Nam3\ML\Other\Final-machine-learning\svm_weights_lie_stand.json', 'r') as json_file:
#     weights_lie_stand = json.load(json_file)

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.1, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils 
def detectPose(image):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
   
    Returns:
        feature: is input for model SVM
        H: height of object (human)
        W: width of object (human)
    '''
    # cv2.imwrite('image.png',  image)
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(imageRGB)
    xs = []
    ys = []
    H = 0
    W = 0
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            # print(landmark.x, landmark.y)
            xs.append(landmark.x)
            ys.append(landmark.y)
    # print(ys)
    if len(xs) != 0:
        x_max, x_min = max(np.max(xs), 0), max(np.min(xs), 0)
        y_max, y_min = max(0, np.max(ys)), max(np.min(ys), 0)
        H, W = y_max - y_min, x_max- x_min
    
    feature =  xs + ys
    return feature, H, W

# def kernel(x, x_hat, weight):
#     return (weight['gamma'] * np.dot(x, x_hat) + weight['r']) ** weight['d']

# def predict_svm(X, weight):
#     s = 0
#     for dual, sv in zip(weight['dual_coef'][0], weight['support_vectors']):
#         s += dual * kernel(sv, X, weight)

#     s += weight['bias']
#     return [int(s > 0)], s

def predict_svm(X):
    return loaded_svm_model.predict(np.array(X).reshape(1, -1))

# %%
LABEL = ['SIT', 'STAND', 'LIE']


# %%
import json
def procees_info_video():

    id_length = 0
  
    cnt_frame = 0
    vid_capture = cv2.VideoCapture(0)
    # check video is opened
    if not vid_capture.isOpened():
        print(f"Không thể mở video cammera")
        return 

    # Count frame begin with start_frame
    cnt = 0 
    text = ""
    count_iter = 0
    while vid_capture.isOpened():
        ret, frame = vid_capture.read()
  
        count_iter += 1

        if ret == True:
            cnt_frame+=1
            # Sure that 3 frames process in 100ms
            if cnt % 3 == 0:
                # workflow model 
                cnt = 0
                image = frame.copy()
                # Get feature of an image 
                feature, H, W  = detectPose(image)
                
                if len(feature) == 66:
                    # Predict of a image is 0 or 1. If 0 is sit and 1 (lie and stand)
                    predictions = predict_svm(feature)
                    predictions = predictions[0]
                    
                    # if predictions != 0:
                    #     # Predict of a image is 0 or 1. If result is 0 then stand else lie with value 1
                    #     predictions, s = predict_svm(feature, weights_lie_stand)
                    #     predictions = predictions[0]
                    #     if predictions == 0:
                    #         predictions = 1
                    #     else:
                    #         predictions = 2

                else:
                    predictions = 0
                text = LABEL[predictions] + " " + "predict: " + str(predictions)

                id_length+=1
       
            cnt += 1
            
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # Ghi frame vào video
            cv2.imshow('Frame', frame)
            # Thoát khỏi vòng lặp khi nhấn phím 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
 
    vid_capture.release()
    cv2.destroyAllWindows()

    return True

# %%
procees_info_video()

# %%




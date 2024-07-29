from ultralytics import YOLO
import os
import cv2
import shutil

IMAGE_PATH = "/datashare/HW1/labeled_image_data/images/train/0018fa1f-output_0063.png" # the video to be predicted on
PREDICTION_FOLDER = './test_predictions' # folder where the prediction will be saved
WEIGHTS = 'best.pt' # the weights used for predictions

'''
*******************************************************
predicts on an image, saves the predictions as output.jpg
*******************************************************
'''

def predict():
    model = YOLO(WEIGHTS)

    results = model(IMAGE_PATH, device=0, iou=0.4, augment=True)

    for result in results:
        result.save(f"output.jpg")  # save to disk

if __name__ == "__main__":
    predict()



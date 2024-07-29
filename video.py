from ultralytics import YOLO
import os
import cv2
import shutil

VIDEO_PATH = "/datashare/HW1/ood_video_data/surg_1.mp4" # the video to be predicted on
PREDICTION_FOLDER = './test_predictions' # folder where the prediction frames will be saved
WEIGHTS = 'best.pt' # the weights used for predictions

'''
*******************************************************
predicts on a video, saves the predictions as video.mp4
*******************************************************
'''

def clear_prediction_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f'Folder created: {folder_path}')
    else:
        print(f'Folder already exists: {folder_path}')

        # Delete all contents of the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

        print(f'All contents of the folder {folder_path} have been deleted.')

def extract_fps():
    video_path = f"/datashare/HW1/ood_video_data/surg_1.mp4"

    # Open the video file
    video_temp = cv2.VideoCapture(video_path)

    fps = 0

    # Check if the video file opened successfully
    if not video_temp.isOpened():
        print(f"Error: Could not open video {video_path}")
    else:
        # Get the frame rate
        fps = video_temp.get(cv2.CAP_PROP_FPS)
        print(f"Frame rate of the video: {fps} frames per second")

    # Release the video capture object
    video_temp.release()

    return fps


def classify_video_with_latest_train(video_path, output_path):
    print("*"*25, "prediction on video", "*"*25)

    model = YOLO(WEIGHTS)

    results = model(video_path, stream=True, device=0, iou=0.4, augment=True)

    frame = 1
    for result in results:
        result.save(f"{output_path}/frame_{frame}.jpg")  # save to disk
        frame += 1

    print("*" * 20)

    print("*" * 25, "finished prediction", "*" * 25)

def combine_video(output_path, fps):
    # Get list of images
    images = [img for img in os.listdir(output_path) if img.endswith(".jpg")]

    # Sort images based on the numeric part of the filenames
    def extract_number(filename):
        return int(os.path.splitext(filename)[0].split('_')[-1])

    images.sort(key=extract_number)

    # Read the first image to get dimensions
    first_image_path = os.path.join(output_path, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    video_path = f'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # For .mp4 output
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Iterate through images and write them to the video
    i = 1
    for image in images:
        image_path = os.path.join(output_path, image)
        frame = cv2.imread(image_path)
        video.write(frame)
        print(f"proceced: {i} frames")
        i += 1

    # Release the video writer object
    video.release()

if __name__ == "__main__":
    clear_prediction_folder(PREDICTION_FOLDER)
    fps = extract_fps()
    classify_video_with_latest_train(VIDEO_PATH, PREDICTION_FOLDER)
    combine_video(PREDICTION_FOLDER, fps)
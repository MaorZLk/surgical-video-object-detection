import os
import cv2
from ultralytics import YOLO

def seperate_videos():
    # Define the video file path and the output folder for frames
    print("*" * 25, "separting videos to frames", "*" * 25)
    video_folder = "/datashare/HW1/id_video_data/"

    videos = [video for video in os.listdir(video_folder) if video.endswith(".mp4")]

    for i, video_path in enumerate(videos):
        print("*" * 10, f"video number {i}", "*" * 10)
        video_name = video_path.split(".")[0]
        output_folder = f'./video_frames/{video_name}'

        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Open the video file
        video = cv2.VideoCapture(os.path.join(video_folder, video_path))

        # Check if the video opened successfully
        if not video.isOpened():
            print(f"Error: Could not open video {video_path}")
            exit()

        frame_number = 1
        while True:
            # Read the next frame from the video
            ret, frame = video.read()
            
            # If there are no more frames, break the loop
            if not ret:
                break
            
            # Construct the output path for the current frame
            frame_path = os.path.join(output_folder, f"{video_name}-frame_{frame_number}.jpg")
            
            # Save the frame as an image file
            cv2.imwrite(frame_path, frame)
            
            print(f"Saved {frame_path}")
            frame_number += 1

        # Release the video capture object
        video.release()

        print(f"All frames have been extracted to {output_folder}")


def combine_video():
    video_path = "/datashare/HW1/id_video_data/20_2_24_1.mp4"

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
    else:
        # Get the frame rate
        fps = video.get(cv2.CAP_PROP_FPS)
        print(f"Frame rate of the video: {fps} frames per second")

    # Release the video capture object
    video.release()


    image_folder = './predictions'

    # Get list of images
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

    # Sort images based on the numeric part of the filenames
    def extract_number(filename):
        return int(os.path.splitext(filename)[0])

    images.sort(key=extract_number)

    # Read the first image to get dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    video_path = 'output_video.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # For .mp4 output
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Iterate through images and write them to the video
    i = 1
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)
        print(f"proceced: {i} frames")
        i += 1

    # Release the video writer object
    video.release()

def delete_frames(video_name):
    path = f'./video_frames/{video_name}'
    for img in os.listdir(path):
        os.remove(os.path.join(path, img))


if __name__ == "__main__":
    # combine_video()
    # seperate_videos()

    model = YOLO("yolov8n.pt")
    model.info()
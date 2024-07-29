import os
from ultralytics import YOLO
import shutil
import json
import torch
import math

def classify_video_with_latest_train(train_number):
    print("*"*25, "prediction on video", "*"*25)

    model = YOLO(f"/home/student/HW1/runs/detect/train{train_number}/weights/best.pt")

    video_folder = "/datashare/HW1/id_video_data/"

    videos = [video for video in os.listdir(video_folder) if video.endswith(".mp4")]

    for i, video_path in enumerate(videos):
        print("*" * 10, f"video {i}", "*" * 10)
        # Open the video file
        video_name = video_path.split('.')[0]
        results = model(os.path.join(video_folder, video_path), stream=True, device=0)

        frame = 1
        mean_conf = {}
        for result in results:
            result.save_txt(f"./predictions_text/{video_name}/{video_name}-frame_{frame}.txt")  # save to disk
            boxes = result.boxes  # Boxes object for bounding box outputs
            try:
                value = torch.mean(boxes.conf).item() 
                mean_conf[frame] = 0 if math.isnan(value) else value
            except:
                mean_conf[frame] = 0
            frame += 1

        with open(f'./prediction_mean_conf/{video_name}.json','w') as file:
            json.dump(mean_conf, file)

        print("*" * 20)

    print("*" * 25, "finished prediction", "*" * 25)

def choose_semi_data(percentage_of_semi):

    number_of_train_pictures = len(os.listdir('/datashare/HW1/labeled_image_data/images/train'))
    num_images_to_select = percentage_of_semi * number_of_train_pictures
    video_folder = "./video_frames"

    videos = [video for video in os.listdir(video_folder)]

    for i, video_path in enumerate(videos):
        video_path = os.path.join(video_folder, video_path)
        frames = os.listdir(video_path)
        # selected_frames = random.sample(frames, int(num_images_to_select / 2))

        def sorting_func(name):
            conf = 0
            with open(f'./prediction_mean_conf/{video_path.split('/')[-1]}.json') as file:
                mean_conf = json.load(file)
                frame = name.split('_')[-1].split('.')[0]
                conf = mean_conf[frame]
            return conf
                
        frames.sort(key=sorting_func, reverse=True)

        for frame in frames[:min(len(frames),int(num_images_to_select / 2))]:
            try:
                shutil.move(os.path.join(video_path,frame), './semi_sup_data/train/images')
                shutil.move(os.path.join(f'./predictions_text/{video_path.split('/')[-1]}/{frame.split('.')[0]}.txt'), './semi_sup_data/train/labels')
            except:
                pass
    

def return_files_to_original_folder():
    images_folder = './semi_sup_data/train/images'
    for img in os.listdir(images_folder):
        video_name = img.split("-")[0]
        try:
            shutil.move(os.path.join(images_folder,img), f'./video_frames/{video_name}')
        except shutil.Error as e:
            os.remove(os.path.join(images_folder,img))
            print("video frame already in original folder")
    
    label_folder = './semi_sup_data/train/labels'
    for label in os.listdir(label_folder):
        video_name = label.split("-")[0]
        try:
            shutil.move(os.path.join(label_folder,label), f'./predictions_text/{video_name}')
        except shutil.Error as e:
            os.remove(os.path.join(label_folder,label))
            print("frame label already in original folder")


if __name__ == "__main__":
    # classify_video_with_latest_train('')
    # choose_semi_data(0.1)
    return_files_to_original_folder()

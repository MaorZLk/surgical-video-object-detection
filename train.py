from ultralytics import YOLO
import torch
import video_classification as vc
import video_managment
import os
import shutil

def create_needed_folders(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f'Folder created: {folder_path}')
    else:
        print(f'Folder already exists: {folder_path}')

def copy_files(source_dir, destination_dir):
    for filename in os.listdir(source_dir):
        source_file = os.path.join(source_dir, filename)
        destination_file = os.path.join(destination_dir, filename)
        if os.path.isfile(source_file):
            shutil.copy(source_file, destination_file)

    print(f"All files have been copied from {source_dir} to {destination_dir}")

create_needed_folders('./video_frames')
create_needed_folders('./video_frames/4_2_24_B_2')
create_needed_folders('./video_frames/20_2_24_1')

create_needed_folders('./predictions_text')
create_needed_folders('./predictions_text/4_2_24_B_2')
create_needed_folders('./predictions_text/20_2_24_1')

create_needed_folders('./prediction_mean_conf')

create_needed_folders('./semi_sup_data')
create_needed_folders('./semi_sup_data/train')
create_needed_folders('./semi_sup_data/train/images')
create_needed_folders('./semi_sup_data/train/labels')
create_needed_folders('./semi_sup_data/val')
create_needed_folders('./semi_sup_data/val/images')
create_needed_folders('./semi_sup_data/val/labels')
copy_files('/datashare/HW1/labeled_image_data/images/val', './semi_sup_data/val/images')
copy_files('/datashare/HW1/labeled_image_data/labels/val', './semi_sup_data/val/labels')


vc.return_files_to_original_folder()# clear the semi_sup_data folder
video_managment.seperate_videos() # seperate given videos to frames

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)


cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)

model.info()

dropout = 0.9
freeze = 0.9
semi_percentage = 0.1
latest_train = 2
TOT_LAYERS = 225

# Train the model on labeled data
results = model.train(data="train_images_pathes.yaml", epochs=100, device=0)
model = YOLO(f"/home/student/HW1/runs/detect/train/weights/best.pt")

vc.classify_video_with_latest_train('')
vc.choose_semi_data(semi_percentage)
results = model.train(data="semi_images_pathes.yaml", epochs=30, device=0, 
                      freeze=freeze*TOT_LAYERS, dropout=dropout)
vc.return_files_to_original_folder()


for epoch in range(1,301):
    model = YOLO(f"/home/student/HW1/runs/detect/train{latest_train}/weights/best.pt")
    print('*'*50, f'round number {epoch}', '*' * 50)
    if epoch % 5 == 0:
        semi_percentage *= 4

    if epoch % 10 == 0:
        freeze -= 0.02
        dropout -= 0.02

    if epoch % 50 == 0:
        results = model.train(data="train_images_pathes.yaml", epochs=100, device=0)
        latest_train += 1
        
    vc.classify_video_with_latest_train(f'{latest_train}')
    vc.choose_semi_data(semi_percentage)

    results = model.train(data="semi_images_pathes.yaml", epochs=30, device=0, 
                      freeze=freeze*TOT_LAYERS, dropout=dropout)
    latest_train += 1
    vc.return_files_to_original_folder()


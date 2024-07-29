# Object detection of surgical tools
This project shows the implementation of YOLOv8 object detection model on leg surgery videos.
Using semi-supervised learning the model is trained on prelabeled data, and uses unlabeled videos to increase training set size.

## Installation
1. Clone the repository:

```sh
git clone <repository_url>
cd <repository_name>
```

2. Create a Conda environment and activate it:

```sh
conda create --name surgical-tool-detection python=3.8
conda activate surgical-tool-detection
```

3. Install the required packages:

```sh
pip install -r requirements.txt
```

## Train the model
To train the model:

```sh
python train.py
```

## predict on picture
To predict on picture:
```sh
python predict.py
```

In the ```predict.py``` file you can find 2 parmaeters:
```python
IMAGE_PATH = "/datashare/HW1/labeled_image_data/images/train/0018fa1f-output_0063.png" # the video to be predicted on
WEIGHTS = 'best.pt' # the weights used for predictions
```
Change them how you want.

## predict on video
To predict on picture:
```sh
python video.py
```

In the ```video.py``` file you can find 3 parmaeters:
```python
VIDEO_PATH = "/datashare/HW1/ood_video_data/surg_1.mp4" # the video to be predicted on
PREDICTION_FOLDER = './test_predictions' # folder where the prediction frames will be saved
WEIGHTS = 'best.pt' # the weights used for predictions
```
Change them how you want.

## Our final weights.
If you wish to download our final weights you can
[Click Here](https://github.com/MaorZLk/surgical-video-object-detection/blob/main/best.pt)
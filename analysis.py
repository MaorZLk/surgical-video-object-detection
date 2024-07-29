from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import os

model = YOLO("./second_run/train28/weights/best.pt")  # load a pretrained model (recommended for training)
results = model.train(data="train_images_pathes.yaml", epochs=100, device=0,
                      project='./final_analysis')

df = pd.read_csv('/home/student/HW1/final_analysis/train/results.csv')
# print(df)
print(df.columns)

plt.title("Train Box Loss")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(df['                  epoch'], df['         train/box_loss'])
plt.savefig('train_box_loss.png')
plt.show()
plt.clf()

plt.title("Train Classification Loss")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(df['                  epoch'], df['         train/cls_loss'])
plt.savefig('train_cls_loss.png')
plt.show()
plt.clf()

plt.title("Validation Box Loss")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(df['                  epoch'], df['           val/box_loss'])
plt.savefig('val_box_loss.png')
plt.show()
plt.clf()

plt.title("Validation Classification Loss")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(df['                  epoch'], df['           val/cls_loss'])
plt.savefig('val_cls_loss.png')
plt.show()
plt.clf()

plt.title("mAP graph")
plt.ylabel('mAP')
plt.xlabel('epoch')
plt.plot(df['                  epoch'], df['       metrics/mAP50(B)'])
plt.savefig('mAP_graph.png')

counters = [0,0,0]
label_path = '/datashare/HW1/labeled_image_data/labels'
for path in os.listdir(label_path):
    path = os.path.join(label_path, path)
    for label in os.listdir(path):
        with open(os.path.join(path, label)) as file:
            txt = file.read()
            classification = txt.split('\n')
            for clas in classification:
                if clas != '':
                    counters[int(clas.split(' ')[0])] += 1

print(counters)
labels = ['Empty', 'Tweezers', 'Needle_driver']

# Create a bar graph
fig, ax = plt.subplots()
bars = ax.bar(labels, counters, color=['blue', 'orange', 'green'])

# Add numbers inside the bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval - 5, int(yval), ha='center', va='top', color='white')

# Add labels and title
ax.set_ylabel('Values')
ax.set_title('')

# Show the plot
plt.show()
plt.savefig('bar_graph.png')

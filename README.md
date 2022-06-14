# Weapon(Gun) Detection Using YoloV3

This Inference python file detect Gun from given image using YoloV3 and OpenCV.

There are 7 Major steps in this data analysis :
1. Dataset preparation
2. Set up darknet
3. Modify configuration files
4. Split train/validation dataset
5. Set up COLAB environment
6. Training on COLAB
7. Predict with YOLOv3 using OpenCV


## Dataset Preparation 
Download Labelled Dataset : [Gun](http://www.mediafire.com/file/pvfircmboaelkxc/Gun_data_labeled.zip/file)

The dataset required to train a detector with YOLOv3 contains 2 components: images and labels. 
Each image will be associated with a label file (normally a txt file) which defines the object class and coordinates of object in the image following this syntax: <object-class> <x_center> <y_center> <width> <height>

This images are labelled using labelling tool : [BBox-Label-Tool](https://github.com/puzzledqs/BBox-Label-Tool)

## Set Up Darknet
YOLOv3 was create on Darknet, an open source neural network framework to train detector.

Download Darknet directly using the link above (unzip required) or using Git : [Darknet](https://github.com/pjreddie/darknet)

Next, in Makefile change GPU = 1 ,CUDNN=1 and OPENCV=1. Next, download darknet pre-trained model on Imagenet directly from [Here](https://pjreddie.com/media/files/darknet53.conv.74) (154Mb). 

Finally, copy and overwrite data folder (including images and labels folders) from our Handgun dataset into darknet folder.
  
## Modify Configuration File
In directory darknet\cfg, creating a copy of “yolov3.cfg” in the same folder and renaming it to “yolov3_custom_train.cfg”.
  
  Line 8 & 9: 
  width = 416, height = 416 
  
  Line 20 
  max_batches = 6000 
  
  Line 22 
  steps = 5400 
  
  Line 603, 689, 776: 
  filters = 18
  
  Line 610, 696, 783: 
  classes = 1
  
## Split Train/Validation Dataset

To avoid over-fitting and achieved an objective evaluation regarding our model, we need to split our dataset into training set and validation set. Depending on the number of images in your dataset that our validation set can be extracted from around 5% (small dataset) to 30% (large dataset) of the total dataset. We use a “train.txt” file and “val.txt” to define the directory path of our training and validation images as below:
  
data/images/armas (2976).jpg

data/images/armas (419).jpg

data/images/armas (1416).jpg

data/images/armas (1591).jpg

data/images/armas (1227).jpg

data/images/armas (1509).jpg
  
In the next step, we need to create file “yolo.names” in the directory darknet\data, with object names as each new line. For instance, in our case we need to detect gun only so the first line should be “gun”.

Finally, create file “yolo.data” in the directory darknet\data:

classes= 1 #number of objects, in our case is 1

train  = data/train.txt

valid  = data/val.txt

names = data/yolo.names

backup = backup
  
## Setup COLAB Environment 
 
First I upload dataset and darknet folder on Google Drive. Now my darknet directory contains:

1. images folder in directory darknet\data contains 3000 images
2. labels folder in directory darknet\data contains 3000 .txt file
3. train.txt, val.txt, yolo.data, yolo.names in directory darknet\data
4. yolov3_custom_train.cfg in directory darknet\cfg

Next,I zip darknet folder and upload it on Google Drive. Create a new folder in Drive and name it “backup”. Then access to [Google Collab](https://colab.research.google.com/) and sign in with google account (same account of that Drive which contain darknet.zip).
  
## Training On COLAB
 
Now Train model using this refernce [Training step by step code](https://colab.research.google.com/drive/13-9pAz9nxUYm-0LlNV9tVtS57g8mHAOb)
  
This will give us Weights file and this can be further use for detection. 
 
## Predict With YOLOv3 Using OpenCV
 
Now predict gun in randomly taken image using YOLOv3 , OpenCV , Matplotlib libraries and using Weights and congiguaration file .

Code given in my file [inference demo.ipynb](https://github.com/meetshah2008/Weapon-Detection-/blob/master/inference%20demo.ipynb)

![Screenshot 2022-06-14 175931](https://user-images.githubusercontent.com/76108073/173577624-814b0f91-b03b-4f20-b995-14cd6d9c02cc.png)
![Screenshot 2022-06-14 175846](https://user-images.githubusercontent.com/76108073/173577611-cf3906ab-27ed-4063-a441-c135d7d23801.png)

## Weakness
![Screenshot 2022-06-14 180007](https://user-images.githubusercontent.com/76108073/173577651-74bfd074-e287-4723-a17f-a623bf018ee0.png)

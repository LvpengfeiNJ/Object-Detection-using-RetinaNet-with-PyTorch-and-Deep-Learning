# Object-Detection-using-RetinaNet-with-PyTorch-and-Deep-Learning

In this tutorial, we will learn how to carry out object detection using RetinaNet with PyTorch and deep learning. Basically, we will use a PyTorch pre-trained model that has been on the COCO dataset. We will use the RetinaNet deep learning model to carry out inference in images and videos and analyze the results as well.

## Demo
step 1: 
```
git clone this repo.
```
step 2: 
```
cd RetinaNet
# for object detection in images
  python detect_images.py --input input/image2.jpg --min-size 1200 --threshold 0.5
# python detect_images.py --input input/image2.jpg
# python detect_images.py --input input/image3.jpg
# python detect_images.py --input input/image3.jpg --min-size 1200 --threshold 0.5

# for object detection in videos
python detect_videos.py --input input/video2.mp4

```

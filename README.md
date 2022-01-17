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
![1](./RetinaNet/output/image2_800_t60.jpg)
![2](./RetinaNet/output/video2_800_t60.mp4)


## The input and output format for PyTorch RetinaNet object detection model
The pre-trained RetinaNet model from PyTorch follows almost the same approach for input and output of data as any other pre-trianed PyTorch model for object detection.

It expects an input image of the format '[C, H, W]', that is (channels, height, and width). And we will of course have to provide a batch size as well. This batch size will amount to the number of images in one batch. So, the final input format will be '[N, C, H, W]'. Also, the pixel values of each image should be between '0-1'.

What we need to focus on is the output format from the RetinaNet model. It outputs a list containing a dictionary which in-turn contains the resulting tensors. The format is 'List[Dict[Tensor]]'. The 'Dict' contains the following keys:
- boxes ('FloatTensor[N, 4]'): the predicted boxes in '[x1, y1, x2, y2]' format, with values between '0' and 'H' and '0' and 'W'
- labels ('Int64Tensor[N]'): the predicted labels for each image
- scores ('Tensor[N]'): the scores or each prediction

[ref.](https://debuggercafe.com/object-detection-using-retinanet-with-pytorch-and-deep-learning/)

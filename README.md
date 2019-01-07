# Image-Similarity-using-Deep-Ranking
The goal of this project is to work on the computer vision task of image similarity. Like most tasks in this field, it’s been aided by the ability of deep networks to extract image features.
The task of image similarity is retrieve a set of n images closest to the query image. One application of this task could involve visual search engine where we provide a query image and want to find an image closest that image in the database.
Here we are implementing paper titled "Learning Fine-grained Image Similarity with Deep Ranking”(https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf).

Data Set
For this project, we will use the Tiny ImageNet (https://tiny-imagenet.herokuapp.com/) dataset
(located at /projects/training/bauh/tiny-imagenet-200/). This dataset consists of 200 different classes with 500 images each. The image size is 64x64. The dataset also includes a validation set of 10000 images (with classes).

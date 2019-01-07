# Image-Similarity-using-Deep-Ranking
The goal of this project is to work on the computer vision task of image similarity. Like most tasks in this field, it’s been aided by the ability of deep networks to extract image features.
The task of image similarity is retrieve a set of n images closest to the query image. One application of this task could involve visual search engine where we provide a query image and want to find an image closest that image in the database.
Here we are implementing paper titled "Learning Fine-grained Image Similarity with Deep Ranking”(https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf).

Data Set:- For this project, we will use the Tiny ImageNet (https://tiny-imagenet.herokuapp.com/) dataset
(located at /projects/training/bauh/tiny-imagenet-200/). This dataset consists of 200 different classes with 500 images each. The image size is 64x64. The dataset also includes a validation set of 10000 images (with classes).

Details:- 
Here we will design a simplified version of the deep ranking model as discussed in the paper. Network architecture will look exactly the same, but the details of the triplet sampling layer will be a lot simpler. The architecture consists of 3 identical networks (Q,P,N). Each of these networks take a single image denoted by pi, pi+, pi- respectively.
pi: Input to the Q (Query) network. This image is randomly sampled across any class.
pi+: Input to the P (Positive) network. This image is randomly sampled from the SAME class as the query image.
pi-: Input to the N (Negative) network. This image is randomly sample from any class EXCEPT the class of pi.
The output of each network, denoted by f(pi), f(pi+), f(pi-) is the feature embedding of an image. This gets fed to the ranking layer.

Ranking Layer:- 
The ranking layer just computes the triplet loss. It teaches the network to produce similar feature embeddings for images from the same class (and different embeddings for images from different classes). g is a gap parameter used for regularization purposes.
D is the Euclidean distance between f(pi) and f(pi+/-).
D(p,q)=D(q,p)= (q1−p1)2+(q2−p2)2+...+(qn−pn)2
g is the gap parameter. We use the default value of 1.0, but you can tune it if you’d like (make sure it’s positive).

Testing stage:-
The testing (inference) stage only has one network and accepts only one image. To retrieve the top n similar results of a query image during inference, the following procedure is followed:
1. Compute the feature embedding of the query image.
2. Compare (euclidean distance) the feature embedding of the query image to all the feature
embeddings in the training data (i.e. your database).
3. Rank the results - sort the results based on Euclidean distance of the feature embeddings.
Triplet Sampling Layer
One of the main contributions of the paper is the triplet sampling layer. Sampling the query image (randomly) and the positive sample image (randomly from the same class as the query image) are quite straightforward.
Negative samples are composed of two different types of samples: in-class and out-of-class. For this project, we will implement out-of-class samples only. Again, out-of-class samples are images sampled randomly from any class except the class of the query image.

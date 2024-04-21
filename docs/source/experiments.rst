Experimental setup
------------------

We established several experiments for training and benchmarking our models,
including the Oxford pet experiment and the histology images experiment.


Oxford III Pet
==============

Our first experiment was elaborated on the Oxford-IIIT Pet Dataset. 
This dataset contains 37 classes of pets, with around 200 images per class. 
The dataset is divided into a training set and a test set. 
The training set contains 3680 images, and the test set contains 3669 images. 
The images are of different sizes and aspect ratios. 
The dataset is available at the following link: 
https://www.robots.ox.ac.uk/~vgg/data/pets/

In particular, we used an implementation of this dataset in a simplified and 
easily retriable format, available at the following link:

https://github.com/UN-GCPDS/python-gcpds.image_segmentation


Scorers emulation
=================

On itself, the Oxford-IIIT Pet dataset contains the masks which reffer to the
ground truth and not to labels from different annotators, which makes this 
dataset non suitable for the original intention of the project. However, we
used this dataset to emulate the scorers' behavior, by training previously a model 
with a simple UNet architecture and then using this model to predict for being 
disturbed in the last encoder layer for producing scorers with different lebels of
agreement.

.. image:: resources/oxford_pet_scorers_emulation.png
  :width: 100%
  :align: center
  :alt: How the scorers emulated noisy annotatiosn for the Oxford-IIIT Pet dataset look.

Crowd Seg Histopatological images
=================================

Our second experiment was elaborated on the CrowdSeg dataset, which consists of 
Triple Negative Breast Cancer images labeled by 20 medical students.

This dataset fairly represents the original intention of the project, which is to
provide a tool for pathologists to segment histopathological images.

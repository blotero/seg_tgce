####################
 Experimental setup
####################

We established several experiments for training and benchmarking our
models, including the Oxford pet experiment and the histology images
experiment.

****************
 Oxford III Pet
****************

Our first experiment was elaborated on the Oxford-IIIT Pet Dataset. This
dataset contains 37 classes of pets, with around 200 images per class.
The dataset is divided into a training set and a test set. The training
set contains 3680 images, and the test set contains 3669 images. The
images are of different sizes and aspect ratios. The dataset is
available at the following link:
https://www.robots.ox.ac.uk/~vgg/data/pets/

In particular, we used an implementation of this dataset in a simplified
and easily retriable format, available at the following link:

https://github.com/UN-GCPDS/python-gcpds.image_segmentation

Scorers emulation
=================

On itself, the Oxford-IIIT Pet dataset contains the masks which reffer
to the ground truth and not to labels from different annotators, which
makes this dataset non suitable for the original intention of the
project. However, we used this dataset to emulate the scorers' behavior,
by training previously a model with a simple UNet architecture and then
using this model to predict for being disturbed in the last encoder
layer for producing scorers with different lebels of agreement.

.. image:: resources/oxford_pet_scorers_emulation.png
   :width: 100%
   :align: center
   :alt: How the scorers emulated noisy annotatiosn for the Oxford-IIIT Pet dataset look.

***********************************
 Crowd Seg Histopatological images
***********************************

Our second experiment was elaborated on the CrowdSeg dataset, which
consists of Triple Negative Breast Cancer images labeled by 20 medical
students.

This dataset fairly represents the original intention of the project,
which is to provide a tool for pathologists to segment histopathological
images.

The dataset is conformed by several histology patches of size 512x512
px. Masks labels exits for an expert pathologist and 20 medical
students. Every single patch contains label for every annotator as shown
in the figure:

.. image:: resources/crowd-seg-example-instances.png
   :width: 100%
   :align: center
   :alt: Different labeling instances for three different patches of the CrowdSeg dataset.

Loading the dataset
===================

If you already have a downloaded dataset in a certain directory, you can
load it symply as a keras sequence with the ``ImageDataGenerator``
class:

.. code:: python

   from seg_tgce.data.crowd_seg import ImageDataGenerator

   val_gen = ImageDataGenerator(
      image_dir="<path to your dataset root>/Histology Data/patches/Val",
      mask_dir="<path to your dataset root>/Histology Data/masks/Val",
      batch_size=16,
      n_classes=6,
    )
   print(f"Train len: {len(val_gen)}")
   print(f"Train masks scorers: {val_gen.n_scorers}")
   print(f"Train masks scorers tags: {val_gen.scorers_tags}")
   val_gen.visualize_sample(
     batch_index=8,
     sample_index=8,
     scorers=["NP8", "NP16", "NP21", "expert"],
   )

The ``ImageDataGenerator`` class is a subclass of the Keras ``Sequence``
class, which allows us to load the dataset in a lazy way. When running
the ``visualize_sample`` method, the generator will load the images and
masks from the disk and display them., with a result similar to the
following

.. image:: resources/crowd-seg-generator-visualization.png
   :width: 100%
   :align: center
   :alt: sample from the CrowdSeg dataset with the ``ImageDataGenerator`` class.

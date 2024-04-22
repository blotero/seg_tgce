..
   semantic-segmentation-tgce documentation master file, created by
   sphinx-quickstart on Sun Dec 10 22:55:05 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

######################################
 Semantic Segmentation TCGE Framework
######################################

Segmentation TGCE is a tool for semantic segmentation of images in the
context of multiple annotators.

This tools performs a segmentation task by learning from different
sources (annotators) and combining them to obtain a better segmentation.
The training algorithm does not require performance metrics from the
annotators as an input, but it learns the annotators' performance from
the data, which behaves as a latent variable and regularizes the
learning process for "learning" more from the most reliable annotators.

Once learnt, the model can be used to predict the segmentation of new
images with a better performance than any of the individual annotators.

#############
 Learn more:
#############

.. toctree::
   :maxdepth: 2

   introduction
   elements/loss
   elements/cnn
   experiments
   contribution

####################
 Indices and tables
####################

-  :ref:`genindex`
-  :ref:`search`

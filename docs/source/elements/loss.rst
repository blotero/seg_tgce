Loss for multiple annotators segmentation
=========================================


In general, a loss function in the context of machine learning is a measure of how well 
a model is able to predict the expected outcome. 

In the context of image segmentation, the expected outcome is a binary mask 
that indicates the location of the object of interest in the image. 
The loss function is used to compare the predicted mask with the ground truth 
mask and calculate a measure of how well the predicted mask matches the 
ground truth mask.


Truncated segmentation generalized cross entropy loss
-----------------------------------------------------


   
Given a $k$ class multiple annotators segmentation problem with a dataset like the following'

.. math::

    \mathbf X \in \mathbb{R}^{W \times H}, \{ \mathbf Y_r \in \{0,1\}^{W \times H \times K} \}_{r=1}^R; \;\; \mathbf {\hat Y} \in [0,1]^{W\times H \times K} = f(\mathbf X)

The segmentation mask function will map input output as follows:

.. math::

    f: \mathbb  R ^{W\times H} \to [0,1]^{W\times H\times K}


$\mathbf Y$ will satisfy the following condition for being a softmax-like representation:

.. math::

    \mathbf Y_r[w,h,:] \mathbf{1} ^ \top _ k = 1; \;\; w \in W, h \in H

Now, let's suppose the existence of an annotators reliability map estimation $\Lambda_r; \; r \in R$;


.. math::

    \bigg\{ \Lambda_r (\mathbf X; \theta ) \in [0,1] ^{W\times H} \bigg\}_{r=1}^R

Then, our $TGCE_{SS}$:


.. math::

    TGCE_{SS}(\mathbf{Y}_r,f(\mathbf X;\theta) | \mathbf{\Lambda}_r (\mathbf X;\theta)) =\mathbb E_{r} \left\{ \mathbb E_{w,h} \left\{ \Lambda_r (\mathbf X; \theta) \circ \mathbb E_k \bigg\{    \mathbf Y_r \circ \bigg( \frac{\mathbf 1 _{W\times H \times K} - f(\mathbf X;\theta) ^{\circ q }}{q} \bigg); k \in K  \bigg\}  + \\ \left(\mathbf 1 _{W \times H } - \Lambda _r (\mathbf X;\theta)\right) \circ \bigg(   \frac{\mathbf 1_{W\times H} - (\frac {1}{k} \mathbf 1_{W\times H})^{\circ q}}{q} \bigg); w \in W, h \in H \right\};r\in R\right\} 


Where $q \in (0,1)$

Total Loss for a given batch holding $N$ samples:

.. math::

    \mathscr{L}\left(\mathbf{Y}_r[n],f(\mathbf X[n];\theta) | \mathbf{\Lambda}_r (\mathbf X[n];\theta)\right)  = \frac{1}{N} \sum_{n}^NTGCE_{SS}(\mathbf{Y}_r[n],f(\mathbf X[n];\theta) | \mathbf{\Lambda}_r (\mathbf X[n];\theta))


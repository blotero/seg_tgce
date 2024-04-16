Contribution
============


If you want to contribute to this project, please consider the following practices
according to your contribution target.


Core
----
The ``core/`` directory contains all functionalities for training and running the
semantic segmentation model. Core behaves like a python module. 
For contributing to this section, first setup your development environment as follows:

.. code-block:: bash

    cd core/
    python -m venv .env
    source .env/bin/activate
    pip install -r requirements.txt
    

When refactoring or adding new features, run tests locally with:

.. code-block:: bash

    pytest .
    
Also, use pylint and mypy for linting code:

.. code-block:: bash

    pylint .
    mypy .

Pylint should score your code 10/10 and mypy should find no issues.

Additionally, for formatting code, you can use isort and black:

.. code-block:: bash

    black .
    isort .


Docs
----
The ``docs/`` directory contains all source files for generating these documentation
pages.


Development environment
^^^^^^^^^^^^^^^^^^^^^^^
Please setup your development environment with venv for python 3.11 as follows


.. code-block:: bash

    cd docs/
    python -m venv .env
    source .env/bin/activate
    pip install -r requirements.txt
    

Once your venv is ready, you can lint your pages after adding new content as follows:

.. code-block:: bash

    rstcheck -r source/
    
If your docs sources are right, you should find an output like the following:
``Success! No issues detected.``


Also, you can locally build doc pages with:

.. code-block:: bash

   make html

   
Notebooks
---------


For setting up a local jupyter notebook, run the following (inside your venv):

.. code-block:: bash

    python -m ipykernel install --user --name=seg_tgce_env
    
Then, open jupyter lab and select the created kern
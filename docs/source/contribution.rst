##############
 Contribution
##############

If you want to contribute to this project, please consider the following
practices according to your contribution target.

******
 Core
******

The ``core/`` directory contains all functionalities for training and
running the semantic segmentation model. Core behaves like a python
module. For contributing to this section, first setup your development
environment as follows:

.. code:: bash

   cd core/
   python -m venv .env
   source .env/bin/activate
   pip install -r requirements.txt

When refactoring or adding new features, run tests locally with:

.. code:: bash

   pytest .

Also, use ``pylint`` and ``mypy`` for linting code:

.. code:: bash

   pylint .
   mypy .

Pylint should score your code 10/10 and mypy should find no issues.

Additionally, for formatting code, you can use ``isort`` and ``black``:

.. code:: bash

   black .
   isort .

******
 Docs
******

The ``docs/`` directory contains all source files for generating these
documentation pages.

Development environment
=======================

Please setup your development environment with ``venv`` for python 3.11
as follows

.. code:: bash

   cd docs/
   python -m venv .env
   source .env/bin/activate
   pip install -r requirements.txt

Once your ``venv`` is ready, you can lint your pages after adding new
content as follows:

.. code:: bash

   rstcheck -r source/

If your docs sources are right, you should find an output like the
following: ``Success! No issues detected.``

Also, you can locally build doc pages with:

.. code:: bash

   make html

Besides, if you want to apply formatting to your docs, you can use
``rstfmt``:

.. code:: bash

   rstfmt -r source/

***********
 Notebooks
***********

For setting up a local jupyter notebook, run the following (inside your
venv):

.. code:: bash

   python -m ipykernel install --user --name=seg_tgce_env

Then, open your preference tool (jupyter lab, vscode viewer, etc) and
select the created kernel.

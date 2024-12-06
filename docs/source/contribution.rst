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
module. For contributing to this section, first make sure you have
`poetry <https://python-poetry.org/docs/>`_ installed in your system,
then, set up your environment as follows:

.. code:: bash

   cd core/
   poetry install
   poetry shell

This will let you with an environment with all dependencies and a shell
session ready to go.

When contributing, run your tets locally with:

.. code:: bash

   pytest .

Also, use ``pylint`` and ``mypy`` for linting code. CI pipelines will
run these too and will fail if code quality is not 10/10:

.. code:: bash

   pylint seg_tgce
   mypy seg_tgce

Pylint should score your code 10/10 and mypy should find no issues.

Additionally, for formatting code, you can use ``isort`` and ``black``:

.. code:: bash

   black seg_tgce
   isort --profile=black seg_tgce

Finally, the package can be built and published to pypi with:

.. code:: bash

   poetry build
   poetry publish

******
 Docs
******

The ``docs/`` directory contains all source files for generating these
documentation pages.

Development environment
=======================

Please setup your development environment with ``poetry`` for python
3.11 as follows

.. code:: bash

   cd docs/
   poetry install
   poetry shell

Once your environment is ready, you can lint your pages after adding new
content as follows:

.. code:: bash

   rstcheck -r source/

If your docs sources are right, you should find an output like the
following: ``Success! No issues detected.``

Also, you can locally build doc pages with:

.. code:: bash

   make html

Please apply formatting to your docs for keeping up with the standard
with ``rstfmt``:

.. code:: bash

   rstfmt source/

***********
 Notebooks
***********

For setting up a local jupyter notebook, run the following (inside your
poetry environment):

.. code:: bash

   python -m ipykernel install --user --name=seg_tgce_env

Then, open your preference tool (jupyter lab, vscode viewer, etc) and
select the created kernel.

# Tensorflow robot learning library

Author : Emmanuel Pignat, emmanuel.pignat@gmail.com

## How to install

Is compatible with python 2 and 3, tensorflow 1 and 2

    git clone https://gitlab.idiap.ch/rli/tf_robot_learning.git

    cd tf_robot_learning

    pip install -e .

### Install jupyter kernels

    python2 -m pip install ipykernel
    python2 -m ipykernel install --user

    python3 -m pip install ipykernel
    python3 -m ipykernel install --user

## Notebooks

	cd [...]/tf_robot_learning/notebooks
	jupyter notebook

Then navigate through folders and click on desired notebook.

| Filename | Description |
|----------|-------------|
| gamp/gamp_time_dependant_mvn_discriminator.ipynb | Simplest example. |
| gamp/gamp_time_dependant_dynamics_learning_ensemble.ipynb | Learning NN dynamics with ensemble network.|
| gamp/gamp_panda.ipynb | Acceleration PoE policy on Panda robot.|



### URDF PARSER

The folder *tf_robot_learning/kinematic/utils/urdf_parser_py* is adapted from https://github.com/ros/urdf_parser_py.
The authors of this code are:

- Thomas Moulard - urdfpy implementation, integration
- David Lu - urdf_python implementation, integration
- Kelsey Hawkins - urdf_parser_python implementation, integration
- Antonio El Khoury - bugfixes
- Eric Cousineau - reflection update
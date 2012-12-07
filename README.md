translational-DBN
=================

A variation on Deep Belief Networks that allows labels to be used in pretraining. 
The application is tracing tongue surface contours from ultrasound images, cf.

Fasel, I., and Berry, J. (2010). Deep belief networks for real-time extraction of tongue contours from ultrasound during speech. 20th International Conference on Pattern Recognition (ICPR), pp. 1493-1496. Available at http://www.u.arizona.edu/~jjberry/ICPR.pdf

The code is designed to run on a CUDA-capable GPU using gnumpy, but can be run
without a GPU. See http://www.cs.toronto.edu/~tijmen/gnumpy.html for details.

An example of a translational deep neural network which ties all these modules
together is forthcoming - stay tuned!

Contents
========

deepnet.py: trains stacks of restricted Boltzmann machines.

backprop.py: trains a neural network with backpropagation using conjugate gradient optimization.

loadData.py: loads and formats ultrasound images and trace files for training.


Dependencies
============

gnumpy - if you run on a GPU you will need to install cudamat, otherwise you
         will need npmat.py from http://www.cs.toronto.edu/~ilya/npmat.py

numpy 

scipy version 0.11 or newer. Older versions don't have the minimize wrapper
used in backprop.py. https://github.com/scipy/scipy

opencv with python bindings: http://opencv.willowgarage.com/wiki/ *note this is only used to load images, and could be replaced with PIL

License
=======

    Copyright (C) 2012  Jeff Berry

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.


translational-DBN
=================

A variation on Deep Belief Networks that allows labels to be used in pretraining. 
The application is tracing tongue surface contours from ultrasound images, cf.

Fasel, I., and Berry, J. (2010). Deep belief networks for real-time extraction of tongue contours from ultrasound during speech. 20th International Conference on Pattern Recognition (ICPR), pp. 1493-1496. Available at http://www.u.arizona.edu/~jjberry/ICPR.pdf

TraintDBN.py: a script to train a translational-DBN as described in the above paper to trace tongue surface contours in ultrasound images.

tDBN.py: contains support classes for TraintDBN.py.

loadData.py: loads and formats ultrasound images and trace files for training.

get_activation.py: method 'autotrace' traces and displays test images using a network trained by TraintDBN.py. The class CRFTrainer contains experimental methods to train a set of Conditional Random Fields on sequences of outputs of the tDBN, using CRF++. 

trainCRFs.py & runCRFs.py: support code for the CRFTrainer class in get_activation.py.

Dependencies
============

theano: http://deeplearning.net/software/theano/

numpy, scipy

opencv with python bindings: http://opencv.willowgarage.com/wiki/ *note this is only used to load images, and could be replaced with PIL

CRF++: http://crfpp.googlecode.com/svn/trunk/doc/index.html

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


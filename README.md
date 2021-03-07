# IBB-Face-Recognition
This is the repository for my 3rd IBB assignment. Here you will find a script which will attempt to identify the certain people in the database,

To succesfully run my script you will need to:
- install the folowwing packages for python: 
	- Tensorflow + Keras (Keras comes with TF in pycharm, i don't know how this works in other environments)
	- numpy, scipy, matplotlib, os (i think this one is by default in python), OpenCV for python
- you will need to extract a dataset next to the script, for example in one of my scripts i used the AWE dataset (awe_sample/train and awe_sample/val)


- you will need to have the data folder (with the val, train and test subfodlers) next to the ibb3.py script
- the ibb3.py script reads the images from the folders in data (example 'data/train/...')
- tensorflow and keras work best for python version 3.6-3.8, it did not work for me on 3.9 because it is not supported there

Have fun with my script! :D

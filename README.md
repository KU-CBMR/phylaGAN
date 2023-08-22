# phylaGAN

Two-step generative pipeline using a combination of C-GAN model and autoencoders that can sample microbiome data from different conditions and provide synthetic data representative of the true data and project it onto a common subspace for disease prediction.


Two datasets are used: 1) T2D study by Qin et al., 2012 and 2) Cirrhosis study by Qin et al., 2014. The files CGAN.py and autoencoder.py are the main files for generation task. Architecture is store in architecture.py Relative abundance in OTUs are present in rows for each individual in the files T2D_OTU.csv and Cirr_OTU.csv. The files for prediction are Cluster_split.R and CNN.py.


**Prerequisites**

1.	Python 2.7
2.	CUDA
3.	cuDNN
4.	Conda
5.	TensorFlow
6.	Torch
7.	NumPy pandas 
8.	Keras





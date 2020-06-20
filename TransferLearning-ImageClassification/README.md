<h1 align="center">
 AutoML-TLIC
</h1>

<h3 align="center">
 Automated Machine Learning - Transfer Learning Image Classifier
</h3>

<p align="center">

  <a href="https://www.python.org/">
    <img alt="Language" src="https://img.shields.io/badge/Language-Python-blue">
  </a>

  <a href="https://www.tensorflow.org/">
    <img alt="Platform: Tensorflow" src="https://img.shields.io/badge/Platform-Tensorflow-orange">
  </a>

  <a href="https://keras.io/">
    <img alt="API: Keras" src="https://img.shields.io/badge/API-Keras-red">
  </a>

  <a href="#">
    <img alt="libraries:Numpy, Pandas, Matplotlib" src="https://img.shields.io/badge/Libraries-NumPy,%20Pandas,%20Matplotlib-yellow">
  </a>

  <a href="https://www.linkedin.com/in/eduardo-henrique-vieira-dos-santos-a29b30114/">
    <img alt="Made by Eduardo Henrique Vieira dos Santos" src="https://img.shields.io/badge/Made%20by-Eduardo%20Henrique%20Vieira%20dos%20Santos-blue">
  </a>

  <a href="#">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-blue">
  </a>

</p>

---

### About

AutoML-TLIC is a command line tool that automates the learning tasks for Machine Learning models to deal with image classification problems using Transfer Learning. 

It performs operations such as balancing data using random undersampling/random oversampling, splitting into training and test data based on proportion using random criteria, loading images to memory then resizing them, loading pretrained weights and freezing them, setting fine tunning layer, applying training parameters, performing data augmentation, executing training and saving the model and its weights as HD5 and JSON.

---

### Requirements
```
Python 3.6.6
Tensorflow 1.13.1
Keras 2.3.1
NumPy 1.15
Pandas 0.23.4
Matplotlib 2.2.3
```

---

### Executing
```
python automl-tlic.py [args]
```
#### Example
```
python automl-tlic.py \
-output_file mymodelfile \
-train_file ~/machine-learning/kaggle/blind-detection/input/aptos2019-blindness-detection/train.csv \
-train_dir ~/machine-learning/kaggle/blind-detection/input/aptos2019-blindness-detection/train_images/ \
-dot_file_extension .png \
-source_column id_code \
-target_column diagnosis \
-img_dim 32 \
-show_plots N \
-test_split .20 \
-test_sampling_perclass 50 \
-train_sampling_perclass 250 \
-net_type xception \
-dense_units 128 \
-epochs 100 \
-patience 50 \
-batch_size 250
```

---

### Arguments
```
--help	:prints this.

-output_file	:output network file name without extension.
				Text/String
				e.g.: myoutputfile

-train_file	:train CSV file path.
			Text/String (use \ before spaces)
			e.g: /home/your-user/Documents/your-folder/train.csv

-train_dir	:path to folder with the images.
			Text/String (use \ before spaces)
			e.g: /home/your-user/Documents/your-folder/train-images/

-dot_file_extension	:extension of the training images.
					Text/String (starts with dot)
					e.g.: .png

-source_column	:column in the training CSV that contains 
				the image file names.
				Text/String
				e.g.: id_code

-target_column	:column in the training CSV that contains 
				the output label.
				Text/String
				e.g.: diagnosis

-img_dimension	:side size of the images in pixels 
				converted to a square image.
				Numeric/Int
				e.g.: 32

-show_plots	:show plots of data and training statistics 
			(ALERT: must close each window to proceed)
			Text/Boolean (N/Y)
			e.g.: N

-test_split	:train test split proportion (default 20\%)
			Numeric/Float
			e.g.: 0.2

-train_sampling_perclass	:balance oversampling/undersampling 
							based of the sample size with N training samples.
							Numeric/Int
							e.g.:250

-test_sampling_perclass		:balance oversampling/undersampling
							based of the sample with N testing samples.
							Numeric/Int
							e.g.:20

-net_type 	:pretrained weights for transfer learning.
			avaliable options: 
				resnet50, inception_v3, xception, mobilenet, vgg19
			Text/String
			e.g.: vgg19

-dense_units	:amount of dense units between the 
				pretrained weights and the output layer.
				Numeric/Int
				e.g.: 128

-epochs		:amount of training epochs.
			Numeric/Int
			e.g.: 100

-patience	:epochs to stop training and pick the 
			better network if no improvement is seen.
			Numeric/Int
			e.g.:50

-batch_size	:number of samples used by the network to propagate
			Numeric/Int
			e.g.:32

```

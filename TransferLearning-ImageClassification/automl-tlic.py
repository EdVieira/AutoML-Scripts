#encoding:utf-8
import sys
import os
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

def getArg(desiredArg, argsValues, argsKeys):
	try:
		return argsValues[argsKeys.index(desiredArg)] 
	except Exception as e:
		return ''

argsKeys = sys.argv[1::2]
argsValues = sys.argv[2::2]

for i in sys.argv:
	if i == '--help':
		print('its there')
		print("""
usage: python automl-tlic.py [args]
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

Example:
python automl-tlic.py \
-output_file mymodelfile \
-train_file ~/machine-learning/kaggle/blind-detection/input/aptos2019-blindness-detection/train.csv \
-train_dir ~/machine-learning/kaggle/blind-detection/input/aptos2019-blindness-detection/train_images/ \
-dot_file_extension .png \
-source_column id_code \
-target_column diagnosis \
-img_dimension 32 \
-show_plots N \
-test_split .20 \
-test_sampling_perclass 50 \
-train_sampling_perclass 250 \
-net_type xception \
-dense_units 128 \
-epochs 100 \
-patience 50 \
-batch_size 250
""")
		exit()

print('Training parameters...')
#output_file = 'mymodelfile'
output_file = getArg('-output_file', argsValues, argsKeys)

#train_file = '.~/machine-learning/kaggle/blind-detection/input/aptos2019-blindness-detection/train.csv'
train_file = getArg('-train_file', argsValues, argsKeys)

#train_dir = '~/machine-learning/kaggle/blind-detection/input/aptos2019-blindness-detection/train_images'
train_dir = getArg('-train_dir', argsValues, argsKeys)

#source_column = 'id_code'
source_column = getArg('-source_column', argsValues, argsKeys)
#target_column = 'diagnosis'
target_column = getArg('-target_column', argsValues, argsKeys)


#dot_file_extension = '.png'
dot_file_extension = getArg('-dot_file_extension', argsValues, argsKeys)

#random_seed = 0
random_seed = getArg('-random_seed', argsValues, argsKeys)
if (random_seed == ''):
	random_seed = 0
else:
	random_seed = int(random_seed)

#img_dimension = 300
img_dimension = getArg('-img_dimension', argsValues, argsKeys)
if (img_dimension == ''):
	img_dimension = (300,300)
else:
	img_dimension = (int(img_dimension),int(img_dimension))

#show_plots = 'y'
show_plots = getArg('-show_plots', argsValues, argsKeys)

#test_split = .2
test_split = getArg('-test_split', argsValues, argsKeys)
if (test_split == ''):
	test_split = .2
else:
	test_split = float(test_split)

#test_sampling_perclass = 150
test_sampling_perclass = getArg('-test_sampling_perclass', argsValues, argsKeys)
if (test_sampling_perclass == ''):
	test_sampling_perclass = 0
else:
	test_sampling_perclass = int(test_sampling_perclass)

#train_sampling_perclass = 700
train_sampling_perclass = getArg('-train_sampling_perclass', argsValues, argsKeys)
if (train_sampling_perclass == ''):
	train_sampling_perclass = 0
else:
	train_sampling_perclass = int(train_sampling_perclass)
#patience = 30
patience = getArg('-patience', argsValues, argsKeys)
if (patience == ''):
	patience = None
else:
	patience = int(patience)

print('Model hyperparameters...')
# ResNet 0
# VGG19 4
#net_type = 'vgg19'
net_type = argsValues[argsKeys.index('-net_type')]
net_type = getArg('-net_type', argsValues, argsKeys)
if (net_type not in ['resnet50','inception_v3','xception','mobilenet','vgg19']):
	print('-net_type ',['resnet50','inception_v3','xception','mobilenet','vgg19'])
	exit()
#dense_units = 256
dense_units = getArg('-dense_units', argsValues, argsKeys)
if (dense_units == ''):
	dense_units = 128
else:
	dense_units = int(dense_units)
#epochs = 100
epochs = getArg('-epochs', argsValues, argsKeys)
if (epochs == ''):
	epochs = 100
else:
	epochs = int(epochs)
#batch_size = 32
batch_size = getArg('-batch_size', argsValues, argsKeys)
if (batch_size == ''):
	batch_size = 0
else:
	batch_size = int(batch_size)

steps_per_epoch = None


print('Importing modules...')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
import matplotlib.pyplot as plt

np.random.seed(random_seed)

#Read csv
print('Reading CSV...')
train_csv_df = pd.read_csv(train_file)

#Filter NA
print('Filtering NA...')
train_csv_df.dropna(inplace= True)
print('Total samples:'+str(train_csv_df.count()[source_column]))

if show_plots.lower() != 'n':
	#Plot classes distribution
	print('Plotting classes histogram...')
	train_csv_df.plot.hist()
	plt.show()

# Get minor class
#print('Getting minor class...')
#sample_perclass, test_sampling_perclass = train_csv_df[train_csv_df[target_column] == 1].count()[source_column], train_csv_df[train_csv_df[target_column] == 3].count()[source_column]
#print(sample_perclass, test_sampling_perclass)

output_classes, output_classes_list = len(set(train_csv_df[target_column])), list(set(train_csv_df[target_column]))
#print(output_classes,output_classes_list)


if (train_sampling_perclass == 0):
	print('Under sampling per class...')
	train_sampling_perclass = float('+inf')
	for i in output_classes_list:
		count = train_csv_df[train_csv_df[target_column] == i].count()[source_column]
		if count < train_sampling_perclass:
			train_sampling_perclass = count
print('Training samples per class:'+str(train_sampling_perclass))


# Split test by classes
print('Splitting test set by classes...')
#test_sampling_perclass = int(test_sampling_perclass)
test_cl_df = []
for i in range(output_classes):
	test_class_amout = train_csv_df[train_csv_df[target_column] == i].count()[target_column]
	test_class_splitted = int(test_class_amout*test_split)
	print('Class '+str(output_classes_list[i])+' in proportion '+str(test_class_amout)+'*'+str(test_split)+' = '+str(test_class_splitted))
	if (test_class_splitted < 10):
		raise 'Not enough data for class:'+str(output_classes_list[i])
	test_cl_df.append(train_csv_df[train_csv_df[target_column] == i].sample(test_class_splitted, replace=True))


# Balance test
print('Balancing test set using '+str(test_sampling_perclass)+' samples per class...')
test_df = test_cl_df[0].sample(test_sampling_perclass)
for i in range(1,output_classes):
	test_df = pd.concat([test_df, test_cl_df[i].sample(test_sampling_perclass, replace=True)], axis=0)
test_df = test_df.sample(frac=1).reset_index(drop=True)

print('Total samples for testing:'+str(test_df.count()[source_column]))
if show_plots.lower() != 'n':
	# Plot test data histogram
	print('Plotting test data histogram after balancing...')
	test_df.plot.hist()
	plt.show()

# Drop test data from train set
print('Dropping test data from train set...')
train_id_codes = list(set(train_csv_df[source_column]) - set(test_df[source_column]))


# Get train data
print('Getting training data...')
only_train_df = train_csv_df[train_csv_df[source_column] == train_id_codes[0]]
for i in range(1, len(train_id_codes)):
	row = train_csv_df[train_csv_df[source_column] == train_id_codes[i]]
	only_train_df = pd.concat([only_train_df, row], axis=0)

print('Remaining samples for training:'+str(only_train_df.count()[source_column]))
only_train_df.count()[source_column]
if show_plots.lower() != 'n':
	# Plot training data histogram
	print('Plotting training data histogram before balancing...')
	only_train_df.plot.hist()
	plt.show()

#Split train by classes
print('Splitting train data by classes...')
cl_df = []
for i in range(output_classes):
	cl_df.append(only_train_df[only_train_df[target_column] == i].sample(train_sampling_perclass, replace=True))


# Balance merging train classes
print('Balancing data by merging training classes...')
train_df = cl_df[0].sample(train_sampling_perclass, replace=True)
for i in range(1,output_classes):
	train_df = pd.concat([train_df, cl_df[i].sample(train_sampling_perclass, replace=True)], axis=0)
train_df = train_df.sample(frac=1).reset_index(drop=True)

print('Total samples for training:'+str(train_df.count()[source_column]))

if show_plots.lower() != 'n':
	print('Plotting training data histogram after balancing...')
	train_df.plot.hist()
	plt.show()


# Open and convert image
print('Declaring functions for image file manipulation...')
import matplotlib.image as mpimg
from PIL import Image

def imgpil_to_np(path, size=(64,64)):
	img = Image.open(path)
	img = img.resize(size, Image.ANTIALIAS) # resizes image in-place
	np_im = np.array(img)
	img.close()
	return np_im

def showimg(img):
	return plt.imshow(img)


#from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

def prepareImages(df, dimension = (220,220), path = ''):
	# Open X images and label its Y
	print('Opening images and assigning its labels...')
	X = []
	Y = None
	failed = []
	d = dimension
	for j in tqdm(range(df[source_column].count())):
		try:
			if type(X) == type(None):
				X.append(imgpil_to_np(path+'/'+df[source_column].iloc[j]+dot_file_extension, size=dimension))
			else:
				X.append(imgpil_to_np(path+'/'+df[source_column].iloc[j]+dot_file_extension, size=dimension))
		except Exception as e:
			print('failed',path+df[source_column].iloc[j]+dot_file_extension)
			failed.append(df.index[j])
	df.drop(failed, inplace=True)
	X = np.array(X)
	if type(Y) == type(None):
		Y = df[target_column]
	else:
		Y = pd.concat([Y, df[target_column]], axis=0)
	Y = pd.get_dummies(Y, dummy_na=False)
	
	Y = np.array(Y)
	print('\tInput shape:',X.shape)
	print('\tOutput shape:', Y.shape)
	return X, Y

# X to Y
print('Train dataset:')
X_train, Y_train = prepareImages(train_df, dimension = img_dimension, path=train_dir)
print('Test dataset:')
X_test, Y_test = prepareImages(test_df, dimension = img_dimension, path=train_dir)


#Data augmentation
print('Data augmentation ImageDataGenerator instance...')
from keras.preprocessing.image import ImageDataGenerator  #For Image argumentaton

datagen = ImageDataGenerator(
		shear_range=0.2,
		rotation_range=90,
		width_shift_range=0.2,
		height_shift_range=0.2,
		zoom_range=0.2,
		rescale=.3,
		horizontal_flip=True,
		vertical_flip=True)


print('Function for getting the Tansfer Learning Model...')
def get_pretrained_model(net_type = 0):
	if net_type=='resnet50':
		#input_x = keras.engine.input_layer.Input((3,224,224))#theano
		from keras.applications.resnet50 import ResNet50,preprocess_input
		#model = ResNet50(input_tensor=input_x,weights='imagenet',include_top=False)
		model = ResNet50(weights='imagenet',include_top=False)
		func = preprocess_input

	elif net_type == 'inception_v3':
		#input_x = keras.engine.input_layer.Input((3,299,299))#theano
		from keras.applications.inception_v3 import InceptionV3,preprocess_input
		#model = InceptionV3(input_tensor=input_x,weights='imagenet',include_top=False)
		model = InceptionV3(weights='imagenet',include_top=False)
		func = preprocess_input

	elif net_type == 'xception':
		#input_x = keras.engine.input_layer.Input((299,299,3))#tensorflow
		from keras.applications.xception import Xception,preprocess_input
		#model = Xception(input_tensor=input_x,weights='imagenet',include_top=False)
		model = Xception(weights='imagenet',include_top=False)
		func = preprocess_input

	elif net_type == 'mobilenet':
		#input_x = keras.engine.input_layer.Input((299,299,3))#tensorflow
		from keras.applications.mobilenet import MobileNet,preprocess_input
		#model = MobileNet(input_tensor=input_x,weights='imagenet',include_top=False)
		model = MobileNet(weights='imagenet',include_top=False)
		func = preprocess_input

	elif net_type == 'vgg19':
		#input_x = keras.engine.input_layer.Input((3,299,299))#theano
		from keras.applications.vgg19 import VGG19,preprocess_input
		#model = VGG19(input_tensor=input_x,weights='imagenet',include_top=False)
		model = VGG19(weights='imagenet',include_top=False)
		func = preprocess_input

	return model,preprocess_input 

def get_model(n_classes=1, net_type=1, weights='imagenet', optmizer='adam', dense_units = 0):
	print('Loading pre-trained model '+str(net_type)+'...')
	# create the base pre-trained model
	base_model, preprocess_input = get_pretrained_model(net_type)
	#base_model = keras.applications.inception_v3.InceptionV3(weights=weights, include_top=False)
	
	print('Adding output model...')
	# add a global spatial average pooling layer
	x = base_model.output
	x = keras.layers.GlobalAveragePooling2D()(x)
	# let's add a fully-connected layer
	if dense_units > 0:
		x = keras.layers.Dense(dense_units, activation='relu',
			kernel_regularizer=keras.regularizers.l2(l=0.01))(x)
	# and a logistic layer -- let's say we have 2 classes
	predictions = keras.layers.Dense(n_classes, activation='softmax')(x)#activation='softmax')(x)
	
	print('Concatenating...')
	# this is the model we will train
	model = keras.models.Model(inputs=base_model.input, outputs=predictions)
	
	print('Freezing pretrained weights...')
	for layer in base_model.layers:
		layer.trainable = False
		
	print('Compile with '+optmizer+' optimizer...')
	# compile the model (should be done *after* setting layers to non-trainable)
	model.compile(optimizer=optmizer, loss='categorical_crossentropy', metrics=['accuracy'])
	"""Alternatively, you can use the loss function `sparse_categorical_crossentropy` instead,
	which does expect integer targets"""
	return model, preprocess_input

from keras.callbacks import *
def get_callbacks():
	model_early_stop = EarlyStopping(monitor='val_accurracy', verbose=1, patience=patience, mode='max', restore_best_weights=True) 
	model_callbacks_list = [model_early_stop]
	return model_callbacks_list

def show_statistics(model_history):
	print('Plotting...')
	# list all data in history
	print(model_history.history.keys())
	# summarize history for accuracy
	plt.plot(model_history.history['accuracy'])
	plt.plot(model_history.history['val_accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(model_history.history['loss'])
	plt.plot(model_history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()


# New transfer learning model instance
print('Creating transfer learning model instance...')
model, preprocess_input = get_model(n_classes=output_classes, net_type=net_type, optmizer='rmsprop', dense_units=dense_units)


#Properly preprocess for the choosen TransferLearning
print('Pre-preprocess for TransferLearning...')
X_train_p = preprocess_input(X_train)
X_test_p = preprocess_input(X_test)



# Train the model
print('Training the model...')
model_callbacks_list = get_callbacks()
batch_size = 32
if type(epochs)==type(None):
	epochs = int(len(X_train)*0.2)
steps_per_epoch = int(len(X_train) / batch_size)
model_history = None
evaluation = float('-inf')
# First clone
weights_bkp= model.get_weights()

model_history = model.fit_generator(
	datagen.flow(X_train_p, Y_train, batch_size), 
	validation_data = (X_test_p, Y_test), 
	steps_per_epoch = steps_per_epoch,
	epochs = epochs,
	verbose = 1,
	callbacks = model_callbacks_list
)

print('Evaluating...')
print('Training loss/acc')
print(model.evaluate(X_train, Y_train))
print('Test loss/acc')
print(model.evaluate(X_test, Y_test))

if show_plots.lower() != 'n':
	show_statistics(model_history)


print('Saving model...')
model.save(output_file+'.h5') #compile not needed

print('Saving weights...')
model.save_weights(output_file+'.weights.h5') #need compile

print('Saving JSON...')
model_json = model.to_json()
with open(output_file+'.json', 'w') as json_file:
    json_file.write(model_json)

print('end')
# Garbage Image Classification Using Convolutional Neural Network (CNN)

The purpose of this project is to classify the garbage images into 6 different categories using convolutional neural network. <br />
To measure the performance, categorical entropy is selected as the loss function, 
and categorical accuracy is set to be the key performance measurement metric for each epoch. <br />
The precision, recall, and F-1 score, along with ROC&AUC at the last epoch are evaluated. <br />

## Data set description

The raw data is downloaded from Kaggle.com (https://www.kaggle.com/asdasdasasdas/garbage-classification). It is a large folder containing 6 subfolders of 2,527 pictures in total. The name of each subfolder indicates the category of garbage images stored in it - cardboard (403 pictures), glass (501 pictures), metal (410 pictures), paper (594 pictures), plastic (482 pictures) and trash (137 pictures) respectively. All the pictures are color-image in jpg format with a size of 512*384 (pixels). <br />

## Data preprocessing
### Data Set Split for Validation
We split the data by 9:1 to get one training set and one validation set respectively. The image data in the train set will be used to fit the neutral networks, and the data in the validation set will be used to do the validation for each epoch.

### Data Augmentation
In the original data set we only had 2,527 images, after the train-test split the images available for training will be even less. Since the performance of deep learning neural networks often improves with the amount of data available, data augmentation is applied. Data augmentation is a technique to artificially create new training data from existing training data. This is done by applying domain-specific techniques to examples from the training data that create new and different training examples.

In our project, we use ImageDataGenerator class, an image generator in the keras.preprocessing.image module, to augment the original image data. Data manipulation techniques as listed in Table 1 are used to expand images in our train
set, and thus improve the generalization ability of our model.

### Data Normalization
The rgb value data will be normalized for all the images (train and validation sets both) by setting the rescale parameter to 1/255

### Additional Data Preprocess
With the help of flow_from_directory, labels are automatically assigned to the image based on the label of its parent directory (the index for each garbage category can be seen in Table 2). We also resize the image size from original 512*384 (pixel) to 300*300 (pixel), which is the input size for the models used afterwards.

The batch size is set to be 16. The batch size impacts how quickly a model learns and the stability of the learning process. A batch sizes of 32 or smaller is recommended in the neural networks training process.

## CNN Model 
### Model Training
When training the neural network models, we do not use fit function but fit_generator function because the training data will be large. It was no longer applicable to simply use model.fit to read the entire training data into memory, so we need to use model.fit_generator to read in batches.

We set epoch equals to 20 and step per epoch is the train set image size (2276) divided by the batch size (16) we designed in the previous section.

### Measurement of Performance
Multiple measurement of performance is used to evaluate the performance of
models listed afterwards.

Categorical entropy and categorical accuracy are selected as the loss function and
the metrics to be tracked for each epoch.

Other conventional evaluation metrics like precision, recall, and F-1 score are also
used to evaluate the model’s classification ability on each garbage category.

ROC and AUC is plotted to provide a visualized idea of the model’s classification
ability for each category of garbage.

## Model Architecture 
We build an MLP with 2 hidden layers. In the input layer we specify the tensor of the input data as 3D. In the output layer we specify that the input data is divided into six categories to correspond to the number of categories of our original data. In the hidden layer, we choose Relu as the activation function, because Relu calculates and converges quickly, and it can also solve the problem of vanishing gradients. It is currently the most mainstream activation function. Because it is a multi- classification problem, our activation function at the output layer is softmax, and the loss function is categorical_crossentropy corresponding to it. 

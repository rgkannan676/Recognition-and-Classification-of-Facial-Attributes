# Recognition and Classification of Facial Attributes
Created a classifier that can identify and classify 41 types of facial attributes, using an ImageNet pre-trained ResNext50 model provided by PyTorch and CelebA dataset. Multiple data augmentation techniques were used along with the MSE loss function. Obtained a training accuracy of 95.7% and test accuracy of 91.8%.

# Labelled Data Division and Loading
Download dataset: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html , use the images in “img_align_celeba.zip” as well as the attribute labels.

The provided labelled data was divided into 3 sets. The data set is divided such that 80% of the data was used for training, 10% for validation and 10% for testing. The function train_test_split() provided by sklearn.model_selection library. 
The data was loaded using the pandas library to load the data into data frame. This data frame is used to manipulate the labelled data.
PIL library was used to load the image and convert the same to tensor value for processing.

# Model
The model used this project is an image net pretrained ResNext50 (32x4d) model provided by PyTorch.
import torchvision.models as models
```
resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
resnext50_32x4d.fc = nn.Linear(2048, 40) # updating final FC layer.
```
ResNext is a modification over the ResNet by reducing the number of hyper parameters required. ResNext has a parameter called cardinality C, which decides how many times a same transformation is done. The C is equal to 32 for the model I selected. This network uses the repetition strategy of ResNet with the split transform merge strategy of the of the Inception network. During multiple experiments, it was found that ResNext performed well compared to ResNet, i.e why I chose ResNext

The dimensions of the output of the last FC (fully connected layer) is changed to 40 as there is 40 features to predict.

As part of the transfer learning, the first 6 child layers were frozen. This was done by setting the attribute param.requires_grad  = False for these layers and mentioning the same in optimizer. The learning was done using  the last 4 layers. This was done to fasten the training process. 

# Loss Function

The Loss function used for this project is the Mean Squared Error (MSE) Loss. The predicted value for each feature was compared with the corresponding label of the given training image. The PyTorch implementation of MSE loss torch.nn.MSELoss() was used for this project. The main idea of choosing this loss function was to reduce the distance between predicted value and ground truth label of the images. The rate of loss reduction was good, and the accuracy improved smoothly after each epochs.

I tried the following loss function but was not good as MSE
•	L1 Loss -  I implemented  this loss, but the accuracy was less compared to MSE loss. 
•	Binary Cross Entropy Loss -  Added a Sigmoid after the final FC and implemented the BCE loss. But the rate reduction of loss function was very less even after couple of epochs. Accuracy was also low compared to MSE and L1 loss.

The PyTorch provided the functionality to calculate the backward propagation gradient. Loss function provide the function loss.backward() to calculate the gradient.

# Optimizer

The optimizer used in this project was Adamax. The implementation provided by PyTorch optim library torch.optim.Adamax() is used. Adamax is a modification over Adams optimizer. Adamax is more robust to gradient update noise and has better numerical stability. 

Optimizer provide a method to update the weights of the tensor using the function optimizer.step(). This will update the weight using the gradients calculated.

# GPU Setup

GPU is used for doing the tensor computation. In PyTorch framework,  we can send the set the optimizer and tensors to Nvidea GPU using <tensorName>.to(“cuda”) function. GPU hardware increases the computation capacity.

# Image Augmentation Techniques Used

Image augmentation techniques were used to improve the accuracy of the network. Following were the augmentations done.

•Horizontal Flip – The image was flipped in the horizontal direction with the help of image transpose facility provided by the PIL library.

•Image resize and cropping – The image was resized to 220 x 270 to maintain the aspect ratio. This is done using the resize function provided by PIL image. Then the resized image is cropped back to 178x218 to fit the input image tensor. This is done using the crop functionality provided by the PIL library.

# Saving the Checkpoints

PyTorch provide a functionality to save the checkpoints by saving the model parameters, optimizer parameters etc. The functions torch.save() and torch.load() can be used to save and load the checkpoints.

# Deciding the threshold

Since the output of each neuron is continuous value, a threshold was set to convert it to binary. In this project 0 (middle of the two labels 1 and -1) was chosen as the threshold since MSE loss was used. If x>=0 the output is 1 else output is -1

# Accuracy Calculation with respect to epochs

# Training

Training was done for a total of 12 epochs over the training data. Below is the accuracy rate found during the training. The models were saved as checkpoint after each epoch for validation and testing.
![image](https://user-images.githubusercontent.com/29349268/118003722-64e4d600-b37b-11eb-9e2a-1538608ef39b.png)


# Validation
Validation was performed in the 12 saved models to find the model that best fits the validation data. Below is the accuracy rate found during validation.
![image](https://user-images.githubusercontent.com/29349268/118003778-7332f200-b37b-11eb-8462-2164f974a193.png)

From the above validation accuracy, we can see that after the 3rd epoch, the accuracy begins to decrease. From this we can infer that after the 3rd training epoch, the model starts to over fit and is not able to generalize well in the validation data

# Testing
The model after the 3rd  epoch is chosen for testing as it had the highest validation accuracy. The testing was done on the test data and had good accuracy comparable to validation. Thus, the model can generalize on test data.

![image](https://user-images.githubusercontent.com/29349268/118015944-6a481d80-b387-11eb-993b-719d24ec1888.png)


Below table contains the observed average accuracy of each facial attribute during testing.
![image](https://user-images.githubusercontent.com/29349268/118003547-42eb5380-b37b-11eb-876f-ac2cd2b48e02.png)


 ![image](https://user-images.githubusercontent.com/29349268/118003421-27804880-b37b-11eb-99d8-ce5728871e5a.png)

Fig: Above is a graph showing the train vs validation accuracy w.r.t to epoch. We can observe that the validation accuracy starts decreasing after the 3rd epoch.

# Accuracy Checking Function
A function named get_accuracy( scores , labels ) was coded and used to find the accuracy by comparing the predicted value with the ground truth. 
Processing Private Test Data
The model of the 3rd  epoch is loaded and used to process the private data set. The output is converted to a pandas data frame and is saved to a file using pandas function to_csv()

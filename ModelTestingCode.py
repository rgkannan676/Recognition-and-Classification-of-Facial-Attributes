import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import torchvision.models as models
import torch.nn as nn
import PIL
from PIL import Image
from torchvision import transforms
import torch.optim as optim
import numpy as np

#Loading the labelled data into pandas data frame
attribute_label_df = pd.read_csv("./list_attr_celeba.txt", delim_whitespace=True,dtype='object',header=None)
attribute_label_df.head()

print("############################# Start Loading Data ##############################")
#Splitting the labelled data into train, validate and test.
train, temp = train_test_split(attribute_label_df, test_size=0.2,shuffle=False)
validate,test = train_test_split(temp, test_size=0.5,shuffle=False)

print(len(train.index))
print(len(validate.index))
print(len(test.index))
print(train.index[0])
print(validate.index[0])
print(test.index[0])
print("############################# Completed Loading Data ##############################")


#Checking if cuda is present. Assigning cuda to device.
print(torch.cuda.is_available())
if torch.cuda.is_available():
    device=torch.device('cuda')
    print("The device is" +str(device))
print("############################# Assigned Device ##############################")

#To calculate the accuracy persentage of the predicted data.
def get_accuracy( scores , labels ):
    accuracy=torch.FloatTensor(1,40)
    bs=scores.size(0)
    for j in range(0,40):
        correctCnt = 0
        for i in range(0,bs):
            if(scores[i,j]==labels[i,j]):
                correctCnt=correctCnt+1
        accuracy[0,j]=(correctCnt/bs)*100
    return accuracy


total_test_Samples = len(test.index)
startTestIndex=int(test.index[0])
default_bs=50
#Converting the label data from data frame to tensor.
image_label_Totest = torch.Tensor(np.array(pd.DataFrame(test.drop(test.columns[0],axis=1)).values).astype('float64')).float()

#Describing the model architecture to initialise with data from checkpoint
resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
resnext50_32x4d.fc = nn.Linear(2048, 40)
ct = 0
for child in resnext50_32x4d.children():
    ct += 1
    if ct < 6:
        for param in child.parameters():
            param.requires_grad = False
            
resnext50_32x4d.to(device)           

#Loading the checkpoint of the model that gave highest accuracy for the validation test data set.
path_toLoad = "./models/model_2_epoch.pt"
checkpoint = torch.load(path_toLoad)

    
#Initializing the model with the model parameters of the checkpoint.
resnext50_32x4d.load_state_dict(checkpoint['model_state_dict'])
#Setting the model to be in evaluation mode. This will set the batch normalization parameters.
resnext50_32x4d.eval() 

total_Batches = int(total_test_Samples/default_bs)
if(total_test_Samples%default_bs>0):
    total_Batches=total_Batches+1
runningAccuracyOfModel=0
featureAccuracy=torch.DoubleTensor(1,40)

for minibatch in range(0,total_Batches):
    torch.cuda.empty_cache()
    if(minibatch==(total_Batches-1)):
        bs=total_test_Samples%default_bs
    else:
        bs=default_bs

    batch_imageTensor = torch.cuda.FloatTensor(bs,3,218,178)
    minibatch_label = torch.cuda.FloatTensor(bs,40)
    for imageNumber in range((minibatch*default_bs)+startTestIndex,(minibatch*default_bs)+bs+startTestIndex):
        #Loading image and converting to tensor.
        imageName = test[0][imageNumber]
        pil_img = Image.open("./img_align_celeba/"+imageName).convert('RGB')
        batch_imageTensor[imageNumber - ((minibatch*default_bs)+startTestIndex)] = transforms.ToTensor()(pil_img)
        minibatch_label[imageNumber - ((minibatch*default_bs)+startTestIndex)] = image_label_Totest[imageNumber-startTestIndex]

    batch_imageTensor.to(device)
    minibatch_label.to(device)
    #Doing prediction on test data
    scores=resnext50_32x4d(batch_imageTensor)       

    #Applying threshold
    converted_Score=scores.clone()
    converted_Score[converted_Score>=0]=1
    converted_Score[converted_Score<0]=-1

    #Calculating accuracy
    accuracy=get_accuracy(converted_Score,minibatch_label)

    del batch_imageTensor
    del minibatch_label
    torch.cuda.empty_cache()

    print( "accuracy tensor in batch "+str(minibatch) +" is "+ str(accuracy))
    print("Net accuracy of batch "+ str(minibatch) +" is "+ str(torch.sum(accuracy)/40) +" %")
    
    #Calculate running accuracy
    runningAccuracyOfModel+=torch.sum(accuracy)/40
    featureAccuracy+=accuracy

print("@@accuracy of model  = "+str(runningAccuracyOfModel/total_Batches))
print("@@Feature wise accuracy of model = "+str(featureAccuracy/total_Batches))

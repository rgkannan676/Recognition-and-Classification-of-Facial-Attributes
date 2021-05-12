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

#Loading the labelled data into pandas data frame.
attribute_label_df = pd.read_csv("./list_attr_celeba.txt", delim_whitespace=True,dtype='object',header=None)
attribute_label_df.head()

print("############################# Start Loading Data ##############################")
#Splitting the labelled data into train, validate and test.
train, temp = train_test_split(attribute_label_df, test_size=0.2,shuffle=False)
validate,test = train_test_split(temp, test_size=0.5,shuffle=False)

print(len(train.index))
print(len(validate.index))
print(len(test.index))
print("############################# Completed Loading Data ##############################")


#Checking if cuda is present. Assigning cuda to device.
print(torch.cuda.is_available())
if torch.cuda.is_available():
    device=torch.device('cuda')
    print("The device is" +str(device))
print("############################# Assigned Device ##############################")

torch.cuda.empty_cache()   
    
#Loading pretrained ResNext model.
resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
resnext50_32x4d.fc = nn.Linear(2048, 40)

#The first 5 layers of the ResNext is frozen using param.requires_grad = False
ct = 0
for child in resnext50_32x4d.children():
    ct += 1
    if ct < 6:
        for param in child.parameters():
            param.requires_grad = False
            
#Senting the model to GPU           
resnext50_32x4d.to(device)
print("############################# Created Model ##############################")


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


total_training_Samples = len(train.index)
default_bs=50

#Intialing loss function to MSE loss
criterion=nn.MSELoss()

#Initializing the optimizer. Mentioning to process the layers whose requires_grad is True.
optimizer = optim.Adamax(filter(lambda p: p.requires_grad, resnext50_32x4d.parameters()))

#Converting the label data from data frame to tensor.
image_label_ToCompare = torch.Tensor(np.array(pd.DataFrame(train.drop(train.columns[0],axis=1)).values).astype('float64')).float()


for epoch in range(0,150):
    total_Batches = int(total_training_Samples/default_bs) + 1
    for minibatch in range(0,total_Batches):
        torch.cuda.empty_cache()
        if(minibatch==(total_Batches-1)):
            bs=total_training_Samples%default_bs
        else:
            bs=default_bs
        
        batch_imageTensor = torch.cuda.FloatTensor(bs*3,3,218,178)
        minibatch_label = torch.cuda.FloatTensor(bs*3,40)
        for imageNumber in range(minibatch*default_bs,(minibatch*default_bs)+bs):
            
            #Loading Image
            imageName = train[0][imageNumber]
            pil_img = Image.open("./img_align_celeba/"+imageName).convert('RGB')
            #Transform Image to tensor
            batch_imageTensor[imageNumber - (minibatch*default_bs)] = transforms.ToTensor()(pil_img)
            minibatch_label[imageNumber - (minibatch*default_bs)] = image_label_ToCompare[imageNumber]
            #Image augmenataion - horizontal flip.
            pil_img=pil_img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            batch_imageTensor[(imageNumber - (minibatch*default_bs))+bs] = transforms.ToTensor()(pil_img)
            minibatch_label[(imageNumber - (minibatch*default_bs))+bs] = image_label_ToCompare[imageNumber]
            #Image augmenataion - resizing and cropping
            pil_img=pil_img.resize((220,270))
            pil_img=pil_img.crop((21,26,199,244))
            batch_imageTensor[(imageNumber - (minibatch*default_bs))+(2*bs)] = transforms.ToTensor()(pil_img)
            minibatch_label[(imageNumber - (minibatch*default_bs))+(2*bs)] = image_label_ToCompare[imageNumber]
   
        #Senting input to GPU
        batch_imageTensor.to(device)
        #Senting label values to GPU
        minibatch_label.to(device)
        
        #Setting the gradients to 0 to remove all previous calculations.
        optimizer.zero_grad()
        #Predicting in input data
        scores=resnext50_32x4d(batch_imageTensor)
        
        #Calculating the loss
        loss =  criterion(scores,minibatch_label) 
        #Calculating the gradients.
        loss.backward()
        #Updating the weights using the loss gradient.
        optimizer.step()
        
        #Applying threshold to convert contious to discrete values.
        converted_Score=scores.clone()
        converted_Score[converted_Score>=0]=1
        converted_Score[converted_Score<0]=-1
        
        #Calculating accuracy
        accuracy=get_accuracy(converted_Score,minibatch_label)
        
        #deleting tensors from GPU to save memmory
        del batch_imageTensor
        del minibatch_label
        #Clearing GPU cache
        torch.cuda.empty_cache()
        
        #Calculating and logging the accuracy and Loss
        print("@Loss in the epoch "+str(epoch)+" in the minibatch "+str(minibatch)+ " is "+ str(loss.detach().item()))
        print("#Accuracy tensor of features is "+str(accuracy))
        print("$Net Accuracy is"+ str(torch.sum(accuracy)/40) +" %")
    
    #Saving the model parameters of the currect epoch.    
    path_toSave = "./model_"+str(epoch)+"_epoch.pt"
    torch.save({
            'model_state_dict': resnext50_32x4d.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, path_toSave)

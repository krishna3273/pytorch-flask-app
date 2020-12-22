import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io

#Load the pre-trained model

input_size=784
hidden_size=100
num_classes=10

class Model(nn.Module):
    def __init__(self,in_size,hidden_size,num_classes):
        super(Model,self).__init__()
        self.lin1=nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        self.lin2=nn.Linear(hidden_size,num_classes)

    def forward(self,x):
        out=self.lin1(x)
        out=self.relu(out)
        out=self.lin2(out)
        return out

loaded_model=Model(in_size=input_size,hidden_size=hidden_size,num_classes=num_classes)
FILE_NAME='app/mnist_nn.pth'
loaded_model.load_state_dict(torch.load(FILE_NAME))
loaded_model.eval()

def transform_image(image):
    #Transforming input to required form
    transform=transforms.Compose([ transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28,28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,),(0.3081,))])
    image=Image.open(io.BytesIO(image))
    #Unsqueeze to add the 4th dimension of batch as it is a single image 
    temp=transform(image).unsqueeze(0)
    print(temp.numpy().shape)
    return temp

def prediction(image):
    image=image.view(-1,784)
    output=loaded_model(image)
    _,y_pred=torch.max(output,1)
    return y_pred

import argparse
import torch 
import torch.nn as nn 
import torchvision 
from train_utils import transformation_Forchheim_train, transformation_test, LambdaLR
from models import ConvNet, EfficientNet_b0, ResNet18
from train_model import train_model

parser = argparse.ArgumentParser(description = 'Train the classifier main file')
parser.add_argument('--train_data_path', type=str, required=True, help='path to train data folder')
parser.add_argument('--valid_data_path', type=str, required=True, help='help to valid data folder')
parser.add_argument('--test_data_path', type=str, required=True, help='path to test data folder')

#training arguments
parser.add_argument('--epochs', type=int, default = 200, help='number of epochs for training')
parser.add_argument('--decay_epoch', type=int, default=100, help='number of decay epoch for training')
parser.add_argument('--batch_size', type=int, default=8, help ='batch size for training')
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--model_output_path', type=str, default='./weight_saved/',help='Directory to save model weights')
parser.add_argument('--number_of_class', type=int, default=3, help='number of class')

# Continue training from checkpoint
parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='Path to the model checkpoint to continue training')

# Choose experiment
parser.add_argument('--experiment', type=str, default='ResNet18', choices=['ResNet18', 'EfficientNet_b0', 'ConvNet'],
                    help='Choose between experiments [ResNet18,EfficientNet,ConvNet]')
parser.add_argument('--optimizer', type=str, default='Adam', choices = ['SGD', 'Adam'], help = 'chosse optimizer')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if args.experiment == 'ResNet18':
	model = ResNet18(nbr_of_class=args.number_of_class) # added args.
	if args.checkpoint_path is not None:
    		model .load_state_dict(torch.load(args.checkpoint_path,map_location=device), strict=False) # added map location to make work on cpu
elif args.experiment == 'EfficientNet_b0':
	model = EfficientNet_b0(nbr_of_class=args.number_of_class) # added args.
	if args.checkpoint_path is not None:
    		model .load_state_dict(torch.load(args.checkpoint_path,map_location=device), strict=False) # added map location to make work on cpu
elif args.experiment == 'ConvNet': #added condition
	model = ConvNet(nbr_of_class=args.number_of_class) # added args.
	if args.checkpoint_path is not None:
    		model .load_state_dict(torch.load(args.checkpoint_path,map_location=device), strict=False) # added map location to make work on cpu

print("---model loaded---")
model = model.to(device)

train_dataset = torchvision.datasets.ImageFolder(root=args.train_data_path,transform=transformation_Forchheim_train)
valid_dataset =  torchvision.datasets.ImageFolder(root=args.valid_data_path,transform=transformation_test)
test_dataset = torchvision.datasets.ImageFolder(root=args.test_data_path,transform=transformation_test) 

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True) 
print("train dataset made")
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
print("valid dataset made") 
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) 
print("test dataset made")

criterion = nn.CrossEntropyLoss()

if args.optimizer == 'SGD':
	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
if args.optimizer == 'Adam':
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

lr_scheduler_optim = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(n_epochs=args.epochs, decay_start_epoch=args.decay_epoch).step) 

print("train will be launched") 
model_myresnet,model_accuracy,model_loss = train_model(train_loader,valid_loader,model,device = device,saving_path=args.model_output_path,criterions=criterion, optimizer=optimizer,nbr_of_class = args.number_of_class, epochs=args.epochs,checkpoint_epochs=50) #added device argument to function

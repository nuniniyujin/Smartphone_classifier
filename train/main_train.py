import argparse
from train_utils import train_utils
from models import ConvNet, EfficientNet_b0, ResNet18
from train_model import train_model

parser = argparse.ArgumentParser(description = 'Train the classifier main file')
parser.add_argument('--train_data_path', type=str, required=True, help='helo to train data folder')
parser.add_argument('--test_data_path', type=str, required=True, help='helo to test data folder')

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
                    help='Choose between experiments ['ResNet18', 'EfficientNet', 'ConvNet']')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if args.experiment == 'ResNet18':
	model = ResNet18(nbr_of_class=number_of_class)
	if args.checkpoint_path is not None:
    		model .load_state_dict(torch.load(args.checkpoint_path), strict=False)
elif args.experiment == 'EfficientNet_b0':
	model = EfficientNet_b0(nbr_of_class=number_of_class)
	if args.checkpoint_path is not None:
    		model .load_state_dict(torch.load(args.checkpoint_path), strict=False)
else:
	model = ConvNet(nbr_of_class=number_of_class)
	if args.checkpoint_path is not None:
    		model .load_state_dict(torch.load(args.checkpoint_path), strict=False)

model = model.to(device)

train_loader = torch.utils.data.DataLoader(args.train_data_path, batch_size = args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(args.test_data_path, batch_size=args.batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
lr_scheduler_optim = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(num_epochs=args.epochs, decay_epoch=args.decay_epoch).step)


model_myresnet,model_accuracy,model_loss = train_model(train_loader, model,saving_path=args.model_output_path,criterions=criterion, optimizer=optimizer, epochs=num_epochs,checkpoint_epochs=50)

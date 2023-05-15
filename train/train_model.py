import torch
import os.path
import time
from test_model import test_model
from datetime import datetime
    

def train_model(train_loader,valid_loader, model,device,saving_path, criterions, optimizer, start_epoch = 0, nbr_of_class = 3,epochs=25,checkpoint_epochs=50,valid_epochs=5):
    
    if not (os.path.exists(saving_path)):
        print("creating folder for output dataset")
        os.mkdir(saving_path)

    model_accuracy = []
    model_loss =[]
    n_class_correct = [0 for i in range(nbr_of_class)]
    n_class_samples = [0 for i in range(nbr_of_class)]

    n_total_steps = len(train_loader)
    print(n_total_steps)
        
    for epoch in range(start_epoch, start_epoch + epochs):
      n_correct=0
      n_samples=0
      acc=0
      loss_average_epoch = 0

      for i, (images, labels) in enumerate(train_loader):
          images = images.to(device) #loading into GPU
          labels = labels.to(device)
          # Forward pass
          outputs = model(images)

          #computing accuracy
          _, predicted = torch.max(outputs, 1) #we save the prediction class value 
          n_samples += labels.size(0) #for each sample tested we increase the value of sample tested (usefull for accuracy calculation)
          n_correct += (predicted == labels).sum().item()  #everytime we predict the good value we increase score 

          for j in range(images.shape[0]):
              label = labels[j]
              pred = predicted[j]
              if (label == pred):
                  n_class_correct[label] += 1
              n_class_samples[label] += 1

          #compute loss
          myloss = criterions(outputs, labels)
          # Backward and optimize
          loss_average_epoch = loss_average_epoch+myloss.item()
          optimizer.zero_grad()
          myloss.backward()
          optimizer.step()      

          if((i%round(n_total_steps*0.3))==0):
              print(f'Epoch [{epoch+1}/{epochs}], Step[{i+1}/{n_total_steps+1}], Step loss: {myloss.item():.4f}')

      acc = 100.0 * n_correct / n_samples
      loss_average_epoch = loss_average_epoch/n_total_steps
      model_accuracy.append(acc)
      model_loss.append(loss_average_epoch/n_total_steps)

      if (epoch%checkpoint_epochs == 0):
           now = datetime.now()
           dt_string = now.strftime("%d_%m_%Y_%Hh%M")
           torch.save(model.state_dict(), os.path.join(saving_path,f"checkpoint_epoch{epoch+1}_{dt_string}.pth"))
          
      print(f'--- Epoch [{epoch+1}/{epochs}], Epoch avg Loss: {loss_average_epoch:.4f}, accuracy:{acc:.4f} ---')

      if (epoch%valid_epochs==0):
        test_model(valid_loader,model,nbr_of_class,device)
    
    print('Finished Training')
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%Hh%M")
    torch.save(model.state_dict(), os.path.join(saving_path,f"FINAL_checkpoint_epoch{epoch+1}_{dt_string}.pth"))

    return model,model_accuracy,model_loss

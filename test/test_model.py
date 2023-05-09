import torch

def test_model(test_loader,model,nbr_of_class, device):
  with torch.no_grad():
    #initialisation of all the metric
      n_correct = 0
      n_samples = 0
      acc=0
      n_class_correct = [0 for i in range(nbr_of_class)]
      n_class_samples = [0 for i in range(nbr_of_class)]

    # test of CNN we load test images in GPU

      for images, labels in test_loader:
          images = images.to(device)
          labels = labels.to(device)
          outputs = model(images) # we submit the test images to our trained Model
          # max returns (value ,index)
          _, predicted = torch.max(outputs, 1) #we save the prediction class value 
          n_samples += labels.size(0) #for each sample tested we increase the value of sample tested (usefull for accuracy calculation)
          n_correct += (predicted == labels).sum().item()  #everytime we predict the good value we increase score 

          for i in range(images.shape[0]):
              label = labels[i]
              pred = predicted[i]
              if (label == pred):
                  n_class_correct[label] += 1
              n_class_samples[label] += 1

      acc = 100.0 * n_correct / n_samples 
      print(f'Accuracy of the network: {acc} %')

      for i in range(nbr_of_class):
          acc = 100.0 * n_class_correct[i] / n_class_samples[i]
          print(f'Accuracy of {classes[i]}: {acc} %')

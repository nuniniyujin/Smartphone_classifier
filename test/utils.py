import cv2
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os.path 
import cv2
import time
import matplotlib.pyplot as plt
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import torchvision.models as models

def draw_grid(img, line_color=(0,0,0), thickness=1, type_=cv2.LINE_AA, pxstep=50):
    '''(ndarray, 3-tuple, int, int) -> void
    draw gridlines on img
    line_color:
        BGR representation of colour
    thickness:
        line thickness
    type:
        8, 4 or cv2.LINE_AA
    pxstep:
        grid line frequency in pixels
    '''
    x = 224
    y = 224
    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pxstep
    return img


def visualization_classification(path, weight_path, gray_option=False, patchsize_M = 224, patchsize_N=224, groundtruth_label =0.0, patch_saving=True,drawing_option=True, color = (0,0,0), num_class = 3, shift_value = 0):
    '''
    visualization heatmap prediction
    path:
        test image path
    gray_option:
      True if we want to test with gray scale image
    weight_path : 
      weight path
    groundtruth_label :
      GT label of the test image (0 : Iphone11, 1 : SamsungGalaxyA235G, 2 : Huawei MatePro50)
    patch_saving :
      saving wront prediction, low prediction patch
    drawing_option:
      draw grid and save final results
    color : 
      prediction text color on the heatmap (0,0,0) or (255,255,255)
    num_class:
      number of classes
    shift_value :
      to shift image origin grid
    '''
    
    file_name = os.path.basename(path)
    file = os.path.splitext(file_name)
    image_name = file[0]
    image_extension = file[1]
    images= cv2.imread(path)


    if gray_option == True:
        images = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
        print(images.shape[0])
        print(images.shape[1])
    else:
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

    M = patchsize_M #patch size
    N = patchsize_N #patch size 
    x1 = 0
    y1 = 0
    num_class = 3 #classification classes
    m = nn.Softmax(dim=1) #define softmax operations



    if gray_option == True:
      images= images[0:images.shape[0]-(images.shape[0]%N),0:images.shape[1]-(images.shape[1]%M)]
      images = np.stack((images,)*3, axis = -1)
    else:
      if((images.shape[1]-(images.shape[1]%M)+shift_value) > images.shape[1]): #check to avoid shift method to be bigger than the image
            lenght_X = images.shape[1]-M-(images.shape[1]%M)+shift_value
      else:
            lenght_X = images.shape[1]-(images.shape[1]%M)+shift_value

      if((images.shape[0]-(images.shape[0]%N)+shift_value) > images.shape[0]): #check to avoid shift method to be bigger than the image
            lenght_Y = images.shape[0]-N-(images.shape[0]%N)+shift_value
      else:
            lenght_Y = images.shape[0]-(images.shape[0]%N)+shift_value

      images= images[shift_value:lenght_Y,shift_value:lenght_X,:]

    cv2.imwrite(str(shift_value) + '_' + image_name + 'shifted' + '.png',cv2.cvtColor(images, cv2.COLOR_RGB2BGR))

    image_copy = images.copy()

    imgheight = images.shape[0]
    print(imgheight)
    imgwidth = images.shape[1]
    print(imgwidth)
    number_of_colums = imgwidth//patchsize_M
    number_of_rows = imgheight//patchsize_N

    print(number_of_colums,number_of_rows)
    empty_image = np.zeros((imgheight, imgwidth,3))
    table = np.zeros((number_of_rows,number_of_colums))


    #define model
    if num_class == 3: #3 classes (0: iphone11, 1:samsunggalaxyA235G, 2:huaweimatepro50)
      model = models.resnet18(pretrained=True)
      num_ftrs = model.fc.in_features #resnet
      model.fc = nn.Linear(num_ftrs, 3) #resnet
      model.load_state_dict(torch.load(weight_path,map_location=torch.device('cpu')))
    else: #binary classifier
      model = models.resnet18(pretrained=True)
      num_ftrs = model.fc.in_features #resnet
      model.fc = nn.Linear(num_ftrs, 2) #resnet
      model.load_state_dict(torch.load(weight_path))
      print('load model')

    # Define transform function
    transformation = transforms.Compose(
        [transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    # Define a transform to convert the image to tensor
    # Convert the image to PyTorch tensor
    tensor = transformation(images)
    tensor=tensor.unsqueeze(dim=0)



    for y in range(0, imgheight, M):
        for x in range(0, imgwidth,N):
            if (imgheight - y) < M or (imgwidth - x) < N:
                break
            y1 = y + M
            x1 = x + N
            print(x//N,y//M)

            if x1 >= imgwidth and y1 >=imgheight:
                x1 = imgwidth -1
                y1 = imgheight -1

                tiles = image_copy[y:y+M, x:x+N]
                tensor = transformation(tiles)
                tensor=tensor.unsqueeze(dim=0)
                with torch.no_grad():
                    outputs = model(tensor) # we submit the test images to our trained Model
                    prob_outputs = m(outputs) #normalized probability output
                    texttensor =  max(max(prob_outputs)) #select max value 
                    stringtext = str(texttensor.item()) #converting max value into string
                    q, predicted = torch.max(outputs, 1) #we save the prediction class value
                    table[y//M,x//N]=predicted.item()

                    if num_class==3: 
                      if predicted.item()==0.0:
                        empty_image[y:y+M, x:x+N]=(255,int(255-255*float(stringtext)),int(255-255*float(stringtext)))
                      elif predicted.item()==1.0:
                        empty_image[y:y+M, x:x+N]=(int(255-255*float(stringtext)),255,int(255-255*float(stringtext)))   
                      else:
                        empty_image[y:y+M, x:x+N]=(int(255-255*float(stringtext)),int(255-255*float(stringtext)),255)


                    if num_class ==2:
                        if predicted.item()==0.0:
                          empty_image[y:y+M, x:x+N]=(255,int(255-255*float(stringtext)),int(255-255*float(stringtext))) 
                        else: 
                          empty_image[y:y+M, x:x+N]=(int(255-255*float(stringtext)),int(255-255*float(stringtext)),255)

                    if patch_saving == True: 
                      if predicted.item() != groundtruth_label:
                          if gray_option != True:
                            cv2.imwrite(image_name + '_'+ str(predicted.item()) +'_' + str(groundtruth_label) + '_' + str(y) +'_'+  str(x) + '_' + str(round(texttensor.item(),2)) + '.png' , cv2.cvtColor(tiles, cv2.COLOR_RGB2BGR))
                          else:
                            cv2.imwrite(image_name + '_'+ str(predicted.item()) +'_' + str(groundtruth_label) + '_' + str(y) +'_'+  str(x) + '_' + str(round(texttensor.item(),2)) + '.png' , cv2.cvtColor(tiles, cv2.COLOR_BGR2GRAY))
                          print('predicted class :' ,predicted.item())
                          print('ground truth label: ', groundtruth_label)


                      if predicted.item() == groundtruth_label:
                        if round(texttensor.item(),2) <  0.65:
                            if gray_option != True:
                              cv2.imwrite(image_name+ '_'+ str(predicted.item()) +'_' + str(groundtruth_label) + '_' +str(y) +'_'+  str(x) +  '_' + str(round(texttensor.item(),2))  + '.png' , cv2.cvtColor(tiles, cv2.COLOR_RGB2BGR))
                            else:
                              cv2.imwrite(image_name+ '_'+ str(predicted.item()) +'_' + str(groundtruth_label) + '_' +str(y) +'_'+  str(x) +  '_' + str(round(texttensor.item(),2))  + '.png' , cv2.cvtColor(tiles, cv2.COLOR_BGR2GRAY))
                            print('predicted class :' ,predicted.item())

                    cv2.putText(img=empty_image, text=stringtext[:4], org=(x+20,y+35), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.5, color=color,thickness=2)


            elif y1 >= imgheight: # when patch height exceeds the image height
                y1 = imgheight - 1
                tiles = image_copy[y:y+M, x:x+N]
                tensor = transformation(tiles)
                tensor=tensor.unsqueeze(dim=0)
                with torch.no_grad():
                    outputs = model(tensor) # we submit the test images to our trained Model
                    prob_outputs = m(outputs) #normalized probability output
                    texttensor =  max(max(prob_outputs)) #select max value
                    stringtext = str(texttensor.item())  #converting max value into string                       
                    q, predicted = torch.max(outputs, 1) #we save the prediction class value
                    table[y//M,x//N]=predicted.item() 
                    if num_class==3: 
                      if predicted.item()==0.0:
                        empty_image[y:y+M, x:x+N]=(255,int(255-255*float(stringtext)),int(255-255*float(stringtext)))
                      elif predicted.item()==1.0:
                        empty_image[y:y+M, x:x+N]=(int(255-255*float(stringtext)),255,int(255-255*float(stringtext)))   
                      else:
                        empty_image[y:y+M, x:x+N]=(int(255-255*float(stringtext)),int(255-255*float(stringtext)),255)


                    if num_class==2:
                        if predicted.item()==0.0:
                          empty_image[y:y+M, x:x+N]=(255,int(255-255*float(stringtext)),int(255-255*float(stringtext))) 
                        else: 
                          empty_image[y:y+M, x:x+N]=(int(255-255*float(stringtext)),int(255-255*float(stringtext)),255)

                    if patch_saving == True:
                      if predicted.item() != groundtruth_label:
                          if gray_option != True:
                            cv2.imwrite(image_name + '_'+ str(predicted.item()) +'_' + str(groundtruth_label) + '_' + str(y) +'_'+  str(x) + '_' + str(round(texttensor.item(),2)) + '.png' , cv2.cvtColor(tiles, cv2.COLOR_RGB2BGR))
                          else:
                            cv2.imwrite(image_name + '_'+ str(predicted.item()) +'_' + str(groundtruth_label) + '_' + str(y) +'_'+  str(x) + '_' + str(round(texttensor.item(),2)) + '.png' , cv2.cvtColor(tiles, cv2.COLOR_BGR2GRAY))
                          print('predicted class :' ,predicted.item())
                          print('ground truth label: ', groundtruth_label)

                      if predicted.item() == groundtruth_label:
                        if round(texttensor.item(),2) <  0.65:
                            if gray_option != True:
                              cv2.imwrite(image_name+ '_'+ str(predicted.item()) +'_' + str(groundtruth_label) + '_' +str(y) +'_'+  str(x) +  '_' + str(round(texttensor.item(),2))  + '.png' , cv2.cvtColor(tiles, cv2.COLOR_RGB2BGR))
                            else:
                              cv2.imwrite(image_name+ '_'+ str(predicted.item()) +'_' + str(groundtruth_label) + '_' +str(y) +'_'+  str(x) +  '_' + str(round(texttensor.item(),2))  + '.png' , cv2.cvtColor(tiles, cv2.COLOR_BGR2GRAY))
                            print('predicted class :' ,predicted.item())
                            print('ground truth label: ', groundtruth_label)

                    cv2.putText(img=empty_image, text=stringtext[:4], org=(x+20,y+35), fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=1.5, color=color,thickness=2)   


            elif x1 >= imgwidth: # when patch width exceeds the image width
                x1 = imgwidth - 1
                tiles = image_copy[y:y+M, x:x+N]
                tensor = transformation(tiles)
                tensor=tensor.unsqueeze(dim=0)
                with torch.no_grad():
                    outputs = model(tensor) # we submit the test images to our trained Model
                    prob_outputs = m(outputs) #normalized probability output
                    texttensor =  max(max(prob_outputs))  #select max value
                    stringtext = str(texttensor.item()) #converting max value into string              
                    q, predicted = torch.max(outputs, 1) #we save the prediction class value 
                    table[y//M,x//N]=predicted.item()

                    if num_class==3: 
                      if predicted.item()==0.0:
                        empty_image[y:y+M, x:x+N]=(255,int(255-255*float(stringtext)),int(255-255*float(stringtext)))
                      elif predicted.item()==1.0:
                        empty_image[y:y+M, x:x+N]=(int(255-255*float(stringtext)),255,int(255-255*float(stringtext)))   
                      else:
                        empty_image[y:y+M, x:x+N]=(int(255-255*float(stringtext)),int(255-255*float(stringtext)),255)


                    if num_class ==2:
                        if predicted.item()==0.0:
                          empty_image[y:y+M, x:x+N]=(255,int(255-255*float(stringtext)),int(255-255*float(stringtext))) 
                        else: 
                          empty_image[y:y+M, x:x+N]=(int(255-255*float(stringtext)),int(255-255*float(stringtext)),255)

                    if patch_saving ==True:
                      if predicted.item() != groundtruth_label:
                          if gray_option != True:
                            cv2.imwrite(image_name + '_'+ str(predicted.item()) +'_' + str(groundtruth_label) + '_' + str(y) +'_'+  str(x) + '_' + str(round(texttensor.item(),2)) + '.png' , cv2.cvtColor(tiles, cv2.COLOR_RGB2BGR))
                          else:
                            cv2.imwrite(image_name + '_'+ str(predicted.item()) +'_' + str(groundtruth_label) + '_' + str(y) +'_'+  str(x) + '_' + str(round(texttensor.item(),2)) + '.png' , cv2.cvtColor(tiles, cv2.COLOR_BGR2GRAY))
                          print('predicted class :' ,predicted.item())
                          print('ground truth label: ', groundtruth_label)

                      if predicted.item() == groundtruth_label:
                        if round(texttensor.item(),2) <  0.65:
                            if gray_option != True:
                              cv2.imwrite(image_name+ '_'+ str(predicted.item()) +'_' + str(groundtruth_label) + '_' +str(y) +'_'+  str(x) +  '_' + str(round(texttensor.item(),2))  + '.png' , cv2.cvtColor(tiles, cv2.COLOR_RGB2BGR))
                            else:
                              cv2.imwrite(image_name+ '_'+ str(predicted.item()) +'_' + str(groundtruth_label) + '_' +str(y) +'_'+  str(x) +  '_' + str(round(texttensor.item(),2))  + '.png' , cv2.cvtColor(tiles, cv2.COLOR_BGR2GRAY))
                            print('predicted class :' ,predicted.item())
                            print('ground truth label: ', groundtruth_label)

                    cv2.putText(img=empty_image, text=stringtext[:4], org=(x+20,y+35), fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=1.5, color=color,thickness=2)   


            else:
                tiles = image_copy[y:y+M, x:x+N]
                tensor = transformation(tiles)
                tensor=tensor.unsqueeze(dim=0)

                with torch.no_grad():
                    outputs = model(tensor) #we submit the test images to our trained Model
                    prob_outputs = m(outputs) #normalized probability output
                    texttensor =  max(max(prob_outputs)) #select max value
                    stringtext = str(texttensor.item()) #converting max value into string        
                    q, predicted = torch.max(outputs, 1) #we save the prediction class value
                    table[y//M,x//N]=predicted.item()
                    if num_class==3: 
                      if predicted.item()==0.0:
                        empty_image[y:y+M, x:x+N]=(255,int(255-255*float(stringtext)),int(255-255*float(stringtext)))
                      elif predicted.item()==1.0:
                        empty_image[y:y+M, x:x+N]=(int(255-255*float(stringtext)),255,int(255-255*float(stringtext)))   
                      else:
                        empty_image[y:y+M, x:x+N]=(int(255-255*float(stringtext)),int(255-255*float(stringtext)),255)

                    if num_class==2:
                        if predicted.item()==0.0:
                          empty_image[y:y+M, x:x+N]=(255,int(255-255*float(stringtext)),int(255-255*float(stringtext))) 
                        else: 
                          empty_image[y:y+M, x:x+N]=(int(255-255*float(stringtext)),int(255-255*float(stringtext)),255)

                    if patch_saving==True:
                      if predicted.item() != groundtruth_label:
                        if gray_option != True:
                            cv2.imwrite(image_name + '_'+ str(predicted.item()) +'_' + str(groundtruth_label) + '_' + str(y) +'_'+  str(x) + '_' + str(round(texttensor.item(),2)) + '.png' , cv2.cvtColor(tiles, cv2.COLOR_RGB2BGR))
                        else:
                            cv2.imwrite(image_name + '_'+ str(predicted.item()) +'_' + str(groundtruth_label) + '_' + str(y) +'_'+  str(x) + '_' + str(round(texttensor.item(),2)) + '.png' , cv2.cvtColor(tiles, cv2.COLOR_BGR2GRAY))
                        print('predicted class :' ,predicted.item())
                        print('ground truth label: ', groundtruth_label)

                      if predicted.item() == groundtruth_label:
                        if round(texttensor.item(),2) <  0.65:
                            if gray_option != True:
                              cv2.imwrite(image_name+ '_'+ str(predicted.item()) +'_' + str(groundtruth_label) + '_' +str(y) +'_'+  str(x) +  '_' + str(round(texttensor.item(),2))  + '.png' , cv2.cvtColor(tiles, cv2.COLOR_RGB2BGR))
                            else:
                              cv2.imwrite(image_name+ '_'+ str(predicted.item()) +'_' + str(groundtruth_label) + '_' +str(y) +'_'+  str(x) +  '_' + str(round(texttensor.item(),2))  + '.png' , cv2.cvtColor(tiles, cv2.COLOR_BGR2GRAY))
                            #cv2.imwrite(image_name+ '_'+ str(predicted.item()) +'_' + str(groundtruth_label) + '_' +str(y) +'_'+  str(x) +  '_' + str(round(texttensor.item(),2))  + '.png' , cv2.cvtColor(tiles, cv2.COLOR_RGB2BGR))
                            print('predicted class :' ,predicted.item())
                            print('ground truth label: ', groundtruth_label)

                    cv2.putText(img=empty_image, text=stringtext[:4], org=(x+20,y+35), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.5, color=color,thickness=2)   




    draw_grid(images,pxstep=M)
    draw_grid(empty_image,pxstep=N)


    #saving concat image
    if drawing_option == True:
      double_image = np.zeros((imgheight, imgwidth*2,3))
      double_image[0:imgheight,0:imgwidth,0] = images[:,:,2]
      double_image[0:imgheight,0:imgwidth,1] = images[:,:,1]
      double_image[0:imgheight,0:imgwidth,2] = images[:,:,0]
      double_image[0:imgheight,imgwidth:imgwidth*2,:] = empty_image
      cv2.imwrite(str(shift_value) + '_' + image_name + '_' + 'concat' + 'Dxocropped' + '.png', double_image)

    #saving alpha image
      images_gray= cv2.imread(path,cv2.IMREAD_GRAYSCALE) #saving original as grayscale
      images_gray= images_gray[shift_value:empty_image.shape[0]+shift_value,shift_value:empty_image.shape[1]+shift_value]

      rgba = np.zeros((imgheight, imgwidth,4))
      rgba[:, :, 3] = images_gray
      rgba[:, :, 0:3] = empty_image
      cv2.imwrite(str(shift_value) + '_' + image_name + '_' + 'alpha'+'Dxocropped' + '.png', rgba)

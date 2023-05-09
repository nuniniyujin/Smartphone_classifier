from dataset.data_utils import heuristic_quality_criterion
import glob
import cv2
import numpy as np
import os
import random

def generate_dataset_for_forchheim(folder_path, output_path, tiles_M = 224, tiles_N=224, nbr_patch_per_image=100, stride=224, shuffle = True):
    
    phone_path_list = []
    for path in os.listdir(folder_path): 
        if path[-1] == "1":
            phone_path_list.append(os.path.join(folder_path,path))

    print(phone_path_list)

    Strain = 97
    #Sval = 18
    Stest= 28

    number_of_labels=len(phone_path_list)

    if not (os.path.exists(output_path)):
        print("creating folder for output dataset")
        os.mkdir(output_path)
        os.mkdir(os.path.join(output_path,"Test"))
        os.mkdir(os.path.join(output_path,"Train"))
        os.mkdir(os.path.join(output_path,"Valid"))
        for i in range(number_of_labels):
            os.mkdir(os.path.join(output_path,f"Test/{i}"))
            os.mkdir(os.path.join(output_path,f"Train/{i}"))
            os.mkdir(os.path.join(output_path,f"Valid/{i}"))

    
    for j in range(number_of_labels):
        image_index = 0  #counter to go through list of files

        if os.path.exists(phone_path_list[j]):
            image_list = glob.glob(phone_path_list[j]+"/*")
        if shuffle:
            random.shuffle(image_list)
        for nbr_of_images_in_list in range(len(image_list)):
            image = cv2.imread(image_list[image_index])
            print(image_list[image_index])
            filename_extract = os.path.basename(image_list[image_index])
            filename_extract = filename_extract[:-4]

            image_copy = image.copy()
            imgheight = image.shape[0]
            imgwidth = image.shape[1]
            nbr_of_tiles = (int(imgheight/tiles_N))* (int(imgwidth/tiles_M))
            tiles_index = np.zeros((nbr_of_tiles,tiles_M,tiles_N,3))
            heuristic_scores = []
            M = tiles_M
            N = tiles_N
            index_var = 0

            for y in range(0, imgheight, stride):
                for x in range(0, imgwidth, stride): 
                        if (imgheight - y) < M or (imgwidth - x) < N:
                            break
                        y1 = y + M
                        x1 = x + N

                        if x1 >= imgwidth and y1 >=imgheight:
                            x1 = imgwidth -1
                            y1 = imgheight -1

                            tiles = image_copy[y:y+M, x:x+N]
                            tiles_index[index_var,:,:,:] = tiles 
                            heuristic_scores.append(heuristic_quality_criterion(tiles))



                        elif y1 >= imgheight: # when patch height exceeds the image height
                            y1 = imgheight - 1
                            #Crop into patches of size MxN
                            tiles = image_copy[y:y+M, x:x+N]
                            tiles_index[index_var,:,:,:] = tiles 
                            heuristic_scores.append(heuristic_quality_criterion(tiles))

                        elif x1 >= imgwidth: # when patch width exceeds the image width
                            x1 = imgwidth - 1
                            #Crop into patches of size MxN
                            tiles = image_copy[y:y+M, x:x+N]
                            tiles_index[index_var,:,:,:] = tiles 
                            heuristic_scores.append(heuristic_quality_criterion(tiles))

                        else:
                            #Crop into patches of size MxN
                            tiles = image_copy[y:y+M, x:x+N]
                            tiles_index[index_var,:,:,:] = tiles 
                            heuristic_scores.append(heuristic_quality_criterion(tiles))
                        
                        index_var += 1



            s = np.array(heuristic_scores)
            sort_index = np.argsort(s)

            for k in range((nbr_of_tiles-1),(nbr_of_tiles-nbr_patch_per_image-1),-1):
                if image_index < Strain: 
                    cv2.imwrite(output_path+f"/Train/{j}/"+str(filename_extract)+"_"+ str(sort_index[k])+'.png', tiles_index[sort_index[k],:,:,:])
                if image_index >= Strain  and image_index < Strain+Stest: 
                    cv2.imwrite(output_path+f"/Test/{j}/"+str(filename_extract)+"_"+ str(sort_index[k])+'.png', tiles_index[sort_index[k],:,:,:])
                if image_index >= Strain+Stest:
                    cv2.imwrite(output_path+f"/Valid/{j}/"+str(filename_extract)+"_"+ str(sort_index[k])+'.png', tiles_index[sort_index[k],:,:,:])

            image_index += 1
        
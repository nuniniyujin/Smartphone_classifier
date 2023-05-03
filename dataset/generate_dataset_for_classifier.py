import random
import cv2
import os
import glob


def generate_dataset_for_classifier(phone_path_list, output_path, patch_size=224,number_of_train_samples=200,number_of_test_samples=400,flat_patch_discard_rate = 98, var_thresholod=400, stride=224):
    '''
    This function generate same number of random patches for the multiple unpaired labels
    - phone_path_list : path list of input dataset for several phones
    - output_path : path where to save generated dataset
    - patch_size: patch height and witdth size to generated dataset(default is 224)
    - number_of_train_samples : number of patch generated for training phase
    - number_of_test_samples : number of patch generated for test phase
    - flat_patch_discard_rate : purcentage of discard rate for flat patch (default 98) 
    - var_thresholod : threshold of pixel variance value to concider a patch "flat" (default is 400)
    - stride : Stride pixel value between each patch generated (default is 224)
               if stride == patch_size, then there is no stride
    '''

    overlapp_acceptance_threshold = 25

    number_of_labels = len(phone_path_list)

    if not (os.path.exists(output_path)):
        print("creating folder for output dataset")
        os.mkdir(output_path)
        os.mkdir(os.path.join(output_path,"Test"))
        os.mkdir(os.path.join(output_path,"Train"))
        for i in range(number_of_labels):
            os.mkdir(os.path.join(output_path,f"Test/{i}"))
            os.mkdir(os.path.join(output_path,f"Train/{i}"))

    M = patch_size
    N = patch_size

    for j in range(number_of_labels):
        #label0_file list making random shuffle
        #file_path_list =(glob.glob(dataset_path+"/0/*"))
        if os.path.exists(phone_path_list[j]):
            label_0_path_list = glob.glob(phone_path_list[j]+"/*")
        random.shuffle(label_0_path_list)

        sample = 0 #counter to check number of samples
        i = 0 #counter to go through list of files

        #generating train label 0
        while(sample<number_of_train_samples):
            image = cv2.imread(label_0_path_list[i])
            print(label_0_path_list[i])
            filename_extract = os.path.basename(label_0_path_list[i])
            filename_extract = filename_extract[:-4]

            image_copy = image.copy()
            imgheight = image.shape[0]
            imgwidth = image.shape[1]
                
            for y in range(0, imgheight, stride):
                for x in range(0, imgwidth, stride): 
                    if (random.randint(0,100)<overlapp_acceptance_threshold):

                        if (imgheight - y) < M or (imgwidth - x) < N:
                            break
                        y1 = y + M
                        x1 = x + N

                        if x1 >= imgwidth and y1 >=imgheight:
                            x1 = imgwidth -1
                            y1 = imgheight -1

                            tiles = image_copy[y:y+M, x:x+N]
                            var_image = tiles.var()
                            if var_image < var_thresholod and sample <number_of_train_samples:
                                n = random.randint(0,100)
                                if n >= flat_patch_discard_rate :
                                    cv2.imwrite(output_path+f"/train/{j}/"+filename_extract+'_tile_'+str(x)+'_'+str(y)+'.png', tiles)
                                    cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 1)
                                    sample += 1
                            elif sample <number_of_train_samples:
                                    cv2.imwrite(output_path+f"/train/{j}/"+filename_extract+'_tile_'+str(x)+'_'+str(y)+'.png', tiles)
                                    cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 1)
                                    sample += 1

                        elif y1 >= imgheight: # when patch height exceeds the image height
                            y1 = imgheight - 1
                            #Crop into patches of size MxN
                            tiles = image_copy[y:y+M, x:x+N]
                            var_image = tiles.var()
                            if var_image < var_thresholod and sample <number_of_train_samples:
                                n = random.randint(0,100)
                                if n >= flat_patch_discard_rate :
                                    cv2.imwrite(output_path+f"/train/{j}/"+filename_extract+'_tile_'+str(x)+'_'+str(y)+'.png', tiles)
                                    cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 1)
                                    sample += 1
                            elif sample <number_of_train_samples:
                                    cv2.imwrite(output_path+f"/train/{j}/"+filename_extract+'_tile_'+str(x)+'_'+str(y)+'.png', tiles)
                                    cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 1)
                                    sample += 1

                        elif x1 >= imgwidth: # when patch width exceeds the image width
                            x1 = imgwidth - 1
                            #Crop into patches of size MxN
                            tiles = image_copy[y:y+M, x:x+N]
                            var_image = tiles.var()
                            if var_image < var_thresholod and sample <number_of_train_samples:
                                n = random.randint(0,100)
                                if n >= flat_patch_discard_rate :
                                    cv2.imwrite(output_path+f"/train/{j}/"+filename_extract+'_tile_'+str(x)+'_'+str(y)+'.png', tiles)
                                    cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 1)
                                    sample += 1
                            elif sample <number_of_train_samples:
                                    cv2.imwrite(output_path+f"/train/{j}/"+filename_extract+'_tile_'+str(x)+'_'+str(y)+'.png', tiles)
                                    cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 1)
                                    sample += 1

                        else:
                            #Crop into patches of size MxN
                            tiles = image_copy[y:y+M, x:x+N]
                            var_image = tiles.var()
                            if var_image < var_thresholod and sample <number_of_train_samples:
                                n = random.randint(0,100)
                                if n >= flat_patch_discard_rate :
                                    cv2.imwrite(output_path+f"/train/{j}/"+filename_extract+'_tile_'+str(x)+'_'+str(y)+'.png', tiles)
                                    cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 1)
                                    sample += 1
                            elif sample <number_of_train_samples:
                                    cv2.imwrite(output_path+f"/train/{j}/"+filename_extract+'_tile_'+str(x)+'_'+str(y)+'.png', tiles)
                                    cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 1)
                                    sample += 1

            print("train samples being generated :",sample," samples generated among ",number_of_train_samples)
            i += 1

        sample = 0
        #generating test label 0
        while(sample<number_of_test_samples):

            image = cv2.imread(label_0_path_list[i])
            filename_extract = os.path.basename(label_0_path_list[i])
            filename_extract = filename_extract[:-4]

            image_copy = image.copy()
            imgheight = image.shape[0]
            imgwidth = image.shape[1]
                
            for y in range(0, imgheight, stride):
     
                for x in range(0, imgwidth,stride):

                    if (random.randint(0,100)<overlapp_acceptance_threshold):

                        if (imgheight - y) < M or (imgwidth - x) < N:
                            break
                        y1 = y + M
                        x1 = x + N

                        if x1 >= imgwidth and y1 >=imgheight:
                            x1 = imgwidth -1
                            y1 = imgheight -1

                            tiles = image_copy[y:y+M, x:x+N]
                            #cv2.imwrite('saved_patches/'+'tile'+str(x)+'_'+str(y)+'.png', tiles)

                            var_image = tiles.var()
                            if var_image < var_thresholod and sample <number_of_test_samples:
                                n = random.randint(0,100)
                                if n >= flat_patch_discard_rate :
                                    cv2.imwrite(output_path+f"/test/{j}/"+filename_extract+'_tile_'+str(x)+'_'+str(y)+'.png', tiles)
                                    cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 1)
                                    sample += 1
                            elif sample <number_of_test_samples:
                                    cv2.imwrite(output_path+f"/test/{j}/"+filename_extract+'_tile_'+str(x)+'_'+str(y)+'.png', tiles)
                                    cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 1)
                                    sample += 1

                        elif y1 >= imgheight: # when patch height exceeds the image height
                            y1 = imgheight - 1
                            #Crop into patches of size MxN
                            tiles = image_copy[y:y+M, x:x+N]
                            var_image = tiles.var()
                            if var_image < var_thresholod and sample <number_of_test_samples:
                                n = random.randint(0,100)
                                if n >= flat_patch_discard_rate :
                                    cv2.imwrite(output_path+f"/test/{j}/"+filename_extract+'_tile_'+str(x)+'_'+str(y)+'.png', tiles)
                                    cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 1)
                                    sample += 1
                            elif sample <number_of_test_samples:
                                    cv2.imwrite(output_path+f"/test/{j}/"+filename_extract+'_tile_'+str(x)+'_'+str(y)+'.png', tiles)
                                    cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 1)
                                    sample += 1

                        elif x1 >= imgwidth: # when patch width exceeds the image width
                            x1 = imgwidth - 1
                            #Crop into patches of size MxN
                            tiles = image_copy[y:y+M, x:x+N]
                            var_image = tiles.var()
                            if var_image < var_thresholod and sample <number_of_test_samples:
                                n = random.randint(0,100)
                                if n >= flat_patch_discard_rate :
                                    cv2.imwrite(output_path+f"/test/{j}/"+filename_extract+'_tile_'+str(x)+'_'+str(y)+'.png', tiles)
                                    cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 1)
                                    sample += 1
                            elif sample <number_of_test_samples:
                                    cv2.imwrite(output_path+f"/test/{j}/"+filename_extract+'_tile_'+str(x)+'_'+str(y)+'.png', tiles)
                                    cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 1)
                                    sample += 1

                        else:
                            #Crop into patches of size MxN
                            tiles = image_copy[y:y+M, x:x+N]
                            var_image = tiles.var()
                            if var_image < var_thresholod and sample <number_of_test_samples:
                                n = random.randint(0,100)
                                if n >= flat_patch_discard_rate :
                                    cv2.imwrite(output_path+f"/test/{j}/"+filename_extract+'_tile_'+str(x)+'_'+str(y)+'.png', tiles)
                                    cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 1)
                                    sample += 1
                            elif sample <number_of_test_samples:
                                    cv2.imwrite(output_path+f"/test/{j}/"+filename_extract+'_tile_'+str(x)+'_'+str(y)+'.png', tiles)
                                    cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 1)
                                    sample += 1

            print("test samples being generated :",sample," samples generated among ",number_of_test_samples)
            i += 1

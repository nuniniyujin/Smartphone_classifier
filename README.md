#Smartphone classifier

## The Forchheim Image Database for Camera Identification in the Wild, 2020
* [Paper link](https://arxiv.org/pdf/2011.02241.pdf)
* unofficial pytorch implementation


###Dataset 
* [Forchheim dataset](https://www.kaggle.com/datasets/aaravsharma180/forchheim-dataset) 
* Dataset contains total of 27 smartphone devices with 25 different models from 9 brands (Samsung Galaxy A6 for devices 15 and 16, Huawei P9 lite for devices 23 and 25)
* 143 scenes are captured in Forchheim, Germany
* Dataset splits (Scene) : S_train = 97, S_val = 18, S_test =28
* Selecting train/valid/test images into non-overlapping 224 x 224 patch cluster candidates, and sort them by the quality criterion Q(P)
* Top 100 candidates are used as patch clusters for the image

Generating dataset for preprocessing : 

```Python
from dataset.generate_dataset_for_forchheim import generate_dataset_for_forchheim
dataset_path = r"path/of/dataset"
output_path = r"/path/of/preprocessed_dataset"
generate_dataset_for_forchheim(folder_path=dataset_path, output_path=output_path, tiles_M = 224, tiles_N=224, nbr_patch_per_image=100, stride=224, shuffle = True)
```




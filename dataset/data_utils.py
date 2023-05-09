import numpy as np

def heuristic_quality_criterion(patch_image):
    alpha = 0.7
    beta = 4.0
    gamma = np.log(0.01)

    # Calculate the mean and standard deviation in each color channel
    red_mean = np.mean(patch_image[:, :, 0])
    green_mean = np.mean(patch_image[:, :, 1])
    blue_mean = np.mean(patch_image[:, :, 2])

    red_std = np.std(patch_image[:, :, 0])
    green_std = np.std(patch_image[:, :, 1])
    blue_std = np.std(patch_image[:, :, 2])

    quality = alpha*beta*(red_mean-red_mean**2)+(1-alpha)*(1-np.exp(gamma*red_std))
    quality += alpha*beta*(green_mean-green_mean**2)+(1-alpha)*(1-np.exp(gamma*green_mean))
    quality += alpha*beta*(blue_mean-blue_mean**2)+(1-alpha)*(1-np.exp(gamma*blue_mean))
    quality = quality/3

    return quality

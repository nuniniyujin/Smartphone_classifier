from generate_dataset_for_forchheim import generate_dataset_for_forchheim
dataset_path = r"/gpfsscratch/rech/jou/uru89tg/forchheim"
output_path = r"/gpfsscratch/rech/jou/uru89tg/forchheim_preprocessed"
generate_dataset_for_forchheim(folder_path=dataset_path, output_path=output_path, tiles_M = 224, tiles_N=224, nbr_patch_per_image=100, stride=224, shuffle = True)

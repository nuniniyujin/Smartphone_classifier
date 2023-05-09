#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=3
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=72:00:00
#SBATCH --output=/gpfswork/rech/gbr/uru89tg/smartphone-classifier/logs/out/train_%j.txt
#SBATCH --error=/gpfswork/rech/gbr/uru89tg/smartphone-classifier/logs/err/train_%j.txt
#SBATCH --mail-user=thomas.eboli@ens-paris-saclay.fr
# #SBATCH --partition=gpu_p2
#SBATCH -C v100-32g

# Clean modules
module purge

# load module
module load pytorch-gpu/py3/1.13.0

cd /gpfswork/rech/gbr/uru89tg//smartphone-classifier
nvidia-smi
free -m

export TRAIN_NAME=$2
export TRAIN_MODULE=$1

srun python main_train.py --train_data_path "D:\Dataset\preprocessed_forchheim_full\Train" --valid_data_path "D:\Dataset\preprocessed_forchheim_full\Valid" --test_data_path "D:\Dataset\preprocessed_forchheim_full\Test" --epochs 2 --number_of_class 25 --experiment "EfficientNet_b0" --optimizer "Adam"

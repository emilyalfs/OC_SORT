#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=2-00:00:00
#SBATCH --constraint=avx

module load protobuf-python/3.14.0-GCCcore-10.2.0
source ~/virtualenvs/oc-sort/bin/activate
export PYTHONDONTWRITEBYTECODE=1

python3 tools/train.py -f exps/example/marblerats/yolox_x_marblerats_train.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
#!/bin/bash

#SBATCH -A precisionhealth_project2
#SBATCH -p precisionhealth

#SBATCH --job-name=encoder_1a
#SBATCH --mail-user=hauthj@umich.edu
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=7g
#SBATCH --time=36:00:00
#SBATCH --gpus=2



#SBATCH --output=%x-%j.log

# The application(s) to execute along with its input arguments and options:
export LAYER_NAME="encoder_1a"
echo "LAYER_NAME: "$LAYER_NAME
python ./encoder_1a_bayesian.py

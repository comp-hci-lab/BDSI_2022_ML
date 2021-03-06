# Big Data Summer Institute (2022) - Machine Learning Group

## Working with Armis2

ssh into the Armis2:
`ssh uniqname@armis2.arc-ts.umich.edu `

For this project, you will use Armis2 services for storing and working with your data. First, clone the git repository here:
`/home/uniqname`

Finally, any data you want towork with can be saved here:
`/scratch/bdsi_root/bdsi1/uniqname`

## Loading necessary modules:
Since you will be working with GPUs, you would have to load the python and cuda modules
<pre>

#Load the correct Python version (3.8.8) in the terminal:
module load python3.8-anaconda

#Next, load the cuda module:
module load cuda
</pre>

## Setting up Virtual Environment

Create a virtual environment:

<pre>
cd ~
pip install virtualenv
which python3
# Use the path printed here as argument to -p, below is an example
virtualenv -p /usr/bin/python3 BdsiEnv
source BdsiEnv/bin/activate
</pre>

By default everything runs in one default python environment,
but doing this will create a localized python3 environment. Whatever you install
in this environment is hidden to the outside global environment, as if you had
a separate python installation altogether. It's always a good idea
to have individual environments for projects to keep workspaces separate.

Install requirements:

Go to BDSI_2022_ML directory and type:

`pip install -r requirements_lite.txt`



## Downloading Data

The virtualenv you set up will contain the module "gdown" which we will use to download files from [Google Drive](https://drive.google.com/drive/folders/1Y-p0NUCtyVz4pKVxgyB8IQWubjOED497?usp=sharing) onto your scratch directory. First, navigate to the [data](https://github.com/comp-hci-lab/BDSI_2022_ML/tree/main/data) folder in the cloned repository. Next, click on the .zip data files and generate their links, which will contain the FILE ID. Enter the FILE ID in the command:

`gdown https://drive.google.com/uc?id=[FILE ID]`

Next, you will have to unzip the files you just downloaded. Type the command in the linux terminal:

`unzip [FILE_NAME].zip `

## Opening Jupyter Notebook within the virtualenv:

<pre>
pip install ipykernel
# Install jupyter kernel in your virtualenv (make sure it is active first)
python -m ipykernel install --user --name BdsiEnv
# To make sure you have installed it correctly, type in:
jupyter kernelspec list
#You should be able to see it in your available kernels

#No browser:
jupyter notebook --no-browser --port=8894
ssh -N -f -L localhost:8892:localhost:8894 uniqname@armis2.arc-ts.umich.edu
</pre>

## Accessing GPUs in Jupyter Notebook:
In order to access GPUs, make sure you have tensorflow-gpu package in your virtualenv. Next, you need to specify certain settings in the [Armis2 browser page](https://armis2.arc-ts.umich.edu/pun/sys/dashboard/batch_connect/sys/arcts_jupyter_notebook/session_contexts/new) order to access the GPUs:
<pre>
Anaconda Python module from which to run Jupyter : python3.8-anaconda/2021.05
Slurm account: bdsi1
partition: gpu
Number of hours: 2
Number of cores: 1
Memory (GB): 4
Number of GPUs: 1
Module commands: load python3.8-anaconda cudnn cuda
</pre>

With the settings above, click on launch.






# Big Data Summer Institute (2022) - Machine Learning Group

## Setup

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

`pip install -r requirements.txt`


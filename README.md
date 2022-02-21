# hls4ml_dense_relu_merge
Testing the hls4ml optimizer pass for merging the ReLU layer into the Dense layer when ReLU immediately follows Dense---a frequently encountered pattern in NNs.


## Setup Dev Environment
We use `venv` to set up our virtual environment.

If you don't already have `venv`, install:
```
python3 -m pip install --user --upgrade pip

python3 -m pip install --user virtualenv
```

Create a virtual environment in this repo
```
python3 -m venv env
```

and then activate it (before install and using packages in the virtual environment)
```
source env/bin/activate
```

Confirm you're in the virtual environment by checking the path of the Python interpreter:
```
which python
```

as it should be in the env directory:
```
env/bin/python
```

If you wish to leave the virtual environment, deactivate it:
```
deactivate
```

I found that I needed to separately install `graphviz`/`dot` on my system using `sudo apt install graphviz`. pip installing `graphviz` was not sufficient.

Install the requirements:
```
python3 -m pip install -r requirements.txt
```

To upgrade a specific requirement:
```
python3 -m pip install --upgrade REQUIREMENT_NAME
```
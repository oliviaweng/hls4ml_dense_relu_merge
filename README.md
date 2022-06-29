# hls4ml_dense_relu_merge
Testing the hls4ml optimizer pass for merging the ReLU layer into the Dense/Conv2D layers when ReLU immediately follows them---a frequently encountered pattern in neural networks (NNs).


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

To synthesize the models, I use `Vivado HLS 2020.1`.

## Train and synthesize a model
I provide two models: 1) a fully connected NN and 2) a convolutional NN. Both are to be trained on the MNIST dataset. *Note*: Currently the `SkipOptimizers` config option doesn't work in hls4ml, so there isn't a way to directly test turning off the relu merge optimization.

### Fully connected NN
To train the fully-connected network, run
```
python3 train.py -c dense.yaml
```

To convert the model into an hls4ml model and Vivado HLS project, run:
```
python3 convert.py -c dense.yaml
```

### Convolutional NN
To train the CNN, run
```
python3 train.py -c conv.yaml
```

To convert the model into an hls4ml model and Vivado HLS project, run:
```
python3 convert.py -c conv.yaml
```
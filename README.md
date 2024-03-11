# Deep Reinforcement Learning for the Mixed-shelves Picker Routing Problem


## Paper 

Implementation of our paper:


> Neural Combinatorial Optimization on Heterogeneous Graphs. An Application to the Picker Routing Problem in Mixed-Shelves Warehouses <br> 
(ICAPS 2024, accepted)<br>
https://openreview.net/forum?id=BL0DDUfSzk


## Dependencies

* Python>=3.9
* NumPy
* [PyTorch](http://pytorch.org/)>=1.13
* tqdm
* hydra (for config management)
* neptune (for logging, disable by setting instance.debug=true)
* Matplotlib (optional, only for plotting)
* Pillow (optional, only for plotting)
* Pandas



## Install
To run this code, first install this library as a python package by running 

```
pip install -e .
```
in the same directory as the setup.py file


## Repo Structure

```
outputs
│
└───<INSTANCE_TYPE1>
│   │   dataset.pth (Test-Dataset + exact solutions for 20 instances)
│   │
│   └───pretrained
│       │   actor.pt (HAM model trained on that instance)
│       │   config.yaml (Config file used to train the model)
│   
└───<INSTANCE_TYPE2>

rl4mixed
│   trainer.py (used for training, validation and testing)
│   validation.py (defines functions for validation and testing)    
│   baselines.py (defines baselines like POMO, greedy rollout ...)   
│
└───models
│   │   attention_model.py (defines all models used for evaluation)
│   │
│   └───encoder 
│   │   │   comb_encoder.py (defines the HAM encoder (working name comb))
│   │   │   ...
│   └───decoder 
│       │   layer.py (defines the decoder layers)
│       │   decoder.py (defines the different decoder architectures)
│   
└───heuristics (defines the heuristic used for evaluation)
│
└───gurobi (defines the gurobipy model)
│
└───generator (defines sampling strategies like beam search, argmax ...)
│
└───problems (defines the Dataset and the State object of the  MSPRP)
│
└───configs (defines the training configs for the various models and instances)
```




## Train 

This repository uses Hydra for experiment configuration. The configuration files can be found in 'rl4mixed/configs'. To run an experiment (e.g. using a specific model from the model configurations in 'rl4mixed/configs/model') use

```
train-actor instance=25s-12i-50p model=comb train.decode_type=pomo_new
```

Here we train the model 'comb' on instances of type '25s-12i-50p' and overwrite the default decoding strategy to 'pomo_new'. All available settings are defined withing dataclasses in the settings.py file. 

During training, metrics are logged using Neptune. This requires a Neptune account as well as an API key. The Neptune username is expected to be stored in an environment variable NEPTUNE_USER and the API key in NEPTUNE_API_TOKEN. **To disable neptune logging set 'instance.debug=true', i.e.**

```
train-actor instance=25s-12i-50p model=comb train.decode_type=pomo_new instance.debug=true
```

## Test

Pre-trained models can be tested, for example on the datasets provided in the 'outputs' folder, by using the flag 'train.train=false' and by providing a path to the folder containing the pretrained model and the respective configuration file using flag '--config-path', e.g.

```
train-actor train.train=false test.make_new_dataset=false --config-path <ROOT>/outputs/10s-3i-20p/pretrained
```
Note, that a test-dataset for a known instance type is automatically loaded when **test.make_new_dataset** is not set to 'true'. 


### Test data generation
New test-datasets can be generated using 


```
sim-testdata instance=<INSTANCE> test.n_exact_instances=50
```
where \<INSTANCE> must correspond to a configuration in /configs/instance



### Parallelization
One may use the hydra multirun option to parallelize experiments over multiple cuda GPUs. To do so, one can specify multiple parameter values by seperating the with a comma and by using the '--multirun' flag, i.e.

```
train-actor instance=25s-12i-50p model=comb,hybrid,flat --multirun
```

By default, this will select the first 3 GPUs for the execution of one experiment each. Note, that there are currently no parallelization options for using multiple GPUs to train a model on a single instance


## Acknowledgements

This repository includes adaptions of the following repositories:
* https://github.com/wouterkool/attention-learn-to-route
* https://github.com/yd-kwon/MatNet
* https://github.com/ai4co/rl4co
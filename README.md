# Machine-QA
Systems for Automated Context Based Question answering.


Usage
-----

Requirements can be installed with `pipenv install`.

Command | What it does
--------|------------
`make prereq` | Gathers and sets up All requirements
`make glove` | Gets [GloVe zip file](https://nlp.stanford.edu/projects/glove/)
`make datasets` | Gets all datasets.
`make squad` | Gets the [SQuAD dataset files](https://rajpurkar.github.io/SQuAD-explorer/)
`python train.py --model simple_rnn` | Train a model.
`python train.py --help` | Display arguments.
`python measure.py --model simple_rnn --checkpoint logs/simple-rnn-64` | Evaluates the model against the SQuAD official dev dataset
`tensorboard --logdir logs` | You can monitor training sessions from here.


To add a new model simply place it in the models folder. The name is the name of the file. This file must define 3 things.

1. `build` is a function which is provided the Config and must return two dictionaries `input` and `output`
2. `feed_gen` is an infinite generator which is provided the dataframe and the config as key word arguments.
3. `config` must be an object which contains all config variables for the model

See [simple_rnn](models/simple_rnn.py) for an example.


Roadmap
-------

- [x] Obtain Dataset + Prep supports
- [x] Build simple RNN (SRNN) based system
- [x] Train SRNN model
- [x] Evaluate SRNN model
- [ ] Set up demo for SRNN model
- [ ] Add other models
    - [ ] [FastQA](https://arxiv.org/abs/1703.04816)
    - [ ] [FusionNet](https://arxiv.org/abs/1711.07341)


Performance
-----------

Date Time                      | Model                | F1         | em
-------------------------------|----------------------|------------|-----
2018-02-10 16:08:34.660689     | logs/simple_rnn-64   | 6.8241152  | 0.1419111 
2018-02-21 08:51:00.718115     | logs/brahma-715      | 7.6588546  | 0.6527909 

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
- [ ] Evaluate SRNN model
- [ ] Set up demo for SRNN model

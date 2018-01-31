# Machine-QA
Systems for Automated Context Based Question answering.


Usage
-----

Command | What it does
--------|------------
`make prereq` | Gathers and sets up All requirements
`make glove` | Gets [GloVe zip file](https://nlp.stanford.edu/projects/glove/)
`make datasets` | Gets all datasets.
`make squad` | Gets the [SQuAD dataset files](https://rajpurkar.github.io/SQuAD-explorer/)
`python train.py` | Train a model. Edit imports to train a specific model


Roadmap
-------

- [x] Obtain Dataset + Prep supports
- [x] Build simple RNN (SRNN) based system
- [x] Train SRNN model
- [ ] Evaluate SRNN model
- [ ] Set up demo for SRNN model

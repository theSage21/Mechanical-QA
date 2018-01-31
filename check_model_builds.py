from models.simple_rnn import build, config

# Colors
WARN = '\033[93m'
GREEN = '\033[92m'
ENDC = '\033[0m'
FAIL = '\033[91m'


print(WARN, 'Building Simple RNN Model', ENDC, sep='')
inp_dict, out_dict = build(**config.__dict__)
print(GREEN, 'Model builds without errors', ENDC)
print(WARN, '=---=='*10, ENDC, sep='')

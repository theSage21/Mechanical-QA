from models import build_simple_rnn

# Colors
WARN = '\033[93m'
GREEN = '\033[92m'
ENDC = '\033[0m'
FAIL = '\033[91m'


print(WARN, 'Building Simple RNN Model', ENDC, sep='')
build_simple_rnn(batch_size=32,
                 max_c_len=60,
                 max_q_len=40,
                 glove_dim=50,
                 summary_dim=10,
                 reasoning_dim=9)
print(GREEN, 'Model builds without errors', ENDC)
print(WARN, '=---=='*10, ENDC, sep='')

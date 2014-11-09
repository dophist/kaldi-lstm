# LSTM-projected BPTT in Kaldi nnet1
## Diagram
![Diagram](https://raw.githubusercontent.com/dophist/kaldi-lstm/master/misc/LSTM_DIAG_EQUATION.jpg)

## Notes:  
* peephole connection(purple) are diagonal
* output-gate peephole is not recursive
* dashed arrows: adaptive weight, i.e activations of (input gate, forget gate, output gate)

Currently implementation includes two versions:
* standard
* google

Go to sub-directory to get more details.

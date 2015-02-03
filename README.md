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

# FAQ
## Q1. How to decode use LSTM AM?
* Standard version: exactly the same as DNN, feed LSTM nnet into nnet-forward as AM scorer.
* Google version: 
	- convert binary nnet into text format via nnet-copy, and open text nnet with your text editor
	- change "Transmit" component to "TimeShift", keep your <Shift> setup consistent with "--targets-delay" used in nnet-train-lstm-streams
	- edit <LstmProjectedStreams> -> <LstmProjected>, remove <NumStream> tag, now the "google version" is converted to "standard version", and you can perform AM scoring via nnet-forward

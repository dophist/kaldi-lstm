LSTM-projection BPTT in Kaldi nnet1
===
Diagram
---
![Diagram](https://raw.githubusercontent.com/dophist/kaldi-lstm/master/misc/LSTM_DIAG_EQUATION.jpg)

Notes:  
* bold arrow: full connection (full matrix)  
* slim arrow: diagonal connection(vector)  
* purple arrow: peephole connection  
* red arrow: LSTM adaptive gate connection  


First google LSTM paper also introduce a non-recurrent projection layer, which are not used in their LSTM for LVCSR(Interspeech 2014 paper), so I didnot implement this.


Time-shift component serves for the target delay in LSTM.

TODO:  
* Due to limited APIs of CuVector, current code uses both vector and matrix representations for computation. Seems redundant.
* bi-directional LSTM  


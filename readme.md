LSTM-projection implementation in Kaldi nnet1
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
	* Because limited APIs in CuVector, so in current code I use both vector and matrix representations. This seems to be a bit redundant. May add extra API to CuVector later.
	* bi-directional LSTM


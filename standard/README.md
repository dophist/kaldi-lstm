# standard version
* Standard LSTM implementation, using epoch-wise BPTT training algorithm, one utterance a time.
* it is now completely compatible with nnet1 codes:
	- you can simple add the lstm component into nnet-component.{h.cc} as usual, just as simple as that
	- you can train Cross-Entropy model with nnet-train-perutt
	- you can do discriminative sequential training(MMI/MPE/sMBR) with nnet-train-mmi-sequential/nnet-train-mpe-sequential

Try it out.

## Targets Delay
Note that in speech recognition, future acoustic context can be beneficial for system performance: DNN achieves this by splicing both left and right neighbouring frames; In LSTM-RNN, this can be done by delaying the targets(typically by 5).
So I also provide a "pre-processing" time-shift component to achieve targets delay, by advancing the features by 5 frames, see the code for details.

## TODO
* bi-directional LSTM  
* Now both vector and matrix representations are used mixed for computation, seems a little redundent. May add DiffSigmoid, DiffTanh to CuVector to clean it up.

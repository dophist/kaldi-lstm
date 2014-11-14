# standard version
* Standard LSTM implementation, using epoch-wise BPTT training algorithm, one utterance a time.
* it is now completely compatible with nnet1 codes:
	- you can simple add the lstm component into nnet-component.{h.cc} as usual, just as simple as that
	- you can train Cross-Entropy model with nnet-train-perutt
	- you can do discriminative sequential training(MMI/MPE/sMBR) with nnet-train-mmi-sequential/nnet-train-mpe-sequential

Try it out.

Note that for LSTM training in speech recognition, a common trick to make use of future context is to delay the targets(typically by 5),
So I also provide a "pre-processing" time-shift component to achieve targets delay, by advancing the features by 5 frames, see the code for details.

## TODO
* bi-directional LSTM  
* Now both vector and matrix representations are used mixed for computation, seems a little redundent. May add DiffSigmoid, DiffTanh to CuVector to clean it up.
* Standard LSTM is vulnerable to gradient exploding if sequences are long and the date set is big. Currently I implement a modified L2 weight max-norm regularizaiton, may add gradient regularization later to further stablize the training.
* To achieve targets delay, it seems not efficient enough to shift features comparing to targets modification, I may add a kaldi-style "pipe" bin to do "online delay" of targets. But currently, doing time shift on features is just fine and not the performance bottle-neck. 

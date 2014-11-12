# standard version
* Standard LSTM implementation, using epoch-wise BPTT training algorithm, one utterance a time.
* Cross-Entropy training
* Discriminative sequential training(MPE, sMBR)

## codes organisation:
* nnet/bd-nnet-lstm-projected.h: core LSTM algorithms
* nnetbin/bd-nnet-train-lstm-perutt.cc: Cross-Entropy training
* nnetbin/bd-nnet-train-lstm-mpe-sequential.cc: discriminative sequential MPE/sMBR training

## modification to existing nnet1 code
* nnet1::Nnet::Reset() & nnet1::Component::Reset() methods are added, used to reset the LSTM component, when an utterance is done and a new utterance is presented.

## TODO
* bi-directional LSTM  
* Now both vector and matrix representations are used mixed for computation, seems a little redundent. May add DiffSigmoid, DiffTanh to CuVector to clean it up.
* Standard LSTM is vulnerable to gradient exploding if sequences are long and the date set is big. Currently I implement a modified L2 weight max-norm regularizaiton, may add gradient regularization later to further stablize the training.

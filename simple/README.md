# simple version
This is a standard LSTM with projection layer, using epoch-wise BPTT on utterance level.
* Standard LSTM implementation, using epoch-wise BPTT training algorithm, one utterance a time.
* Support Cross-Entropy training and discriminative sequential training(MPE, sMBR)

## binary level tools:
* nnetbin/bd-nnet-train-lstm-perutt.cc: Cross-Entropy training
* nnetbin/bd-nnet-train-lstm-mpe-sequential.cc: discriminative sequential MPE/sMBR training

## modification to existing nnet1 code
* Nnet::Reset() & Component::Reset() methods are added to nnet1::Nnet & nnet1::Component class, which is used to reset the LSTM component when an old utt ends and a new utt begins.

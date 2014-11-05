LSTM-projected BPTT in Kaldi nnet1
===
Diagram
---
![Diagram](https://raw.githubusercontent.com/dophist/kaldi-lstm/master/misc/LSTM_DIAG_EQUATION.jpg)

Notes:  
---
* peephole connection(purple) are diagonal
* output-gate peephole is not recursive
* dashed arrows: adaptive weight, i.e activations of (input gate, forget gate, output gate)

Currently implementation includes two versions:

simple version:
---
This is standard LSTM implementation, using epoch-wise BPTT training algorithm.
Support Cross-Entropy training and discriminative sequential training(MPE, sMBR)

faster version:
---
* standard LSTM with epoch-wised BPTT suffers from gradients exploding:
When long sequence is presented, BPTT time unfolding becomes long, backward pass tends to blow up.
This makes LSTM training in **large dataset** unstable.
Google uses a "batched" BPTT (Tbptt=20), which greatly improves training stability.
Inside an utterance, network states of previous batch are saved and bridged to the next batch as initial history states.
* multiple utterances are processed simultaneously(4 utterances per CPU in Google's setup).
I prefer to call this "multi-stream". This greatly speeds up the training.
Another reason to do this "multi-stream" training is that all RNN algorithm is sequential, 
particularly in epoch-wise BPTT, shuffling can only be done in utterance level, frame-level stochasticity is missing.
Multi-stream training receives updates from different utterances at the same time, which improves stochasticity(we are actually using SGD), 

TODO:  
---
* bi-directional LSTM  
* simple version: now mixes both vector and matrix representations for computation. May add DiffSigmoid, DiffTanh to CuVector to clean it up.
* faster version: add multi-stream discriminative sequential training(MMI, sMBR)
* binary level code clean-up is nearly done. Script level code clean-up is on the way.


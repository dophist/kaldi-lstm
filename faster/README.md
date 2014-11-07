# faster version
To read more about this version, read the Google paper in this directory.

notable modification to standard LSTM training includes:
### batched-BPTT
Standard LSTM with epoch-wised BPTT suffers from gradients exploding:
When long sequence is presented, BPTT time unfolding becomes long, backward pass tends to blow up.
This makes LSTM training in **large dataset** unstable.
Similar to williams' BPTT(h,h-prime), Google uses a "batched" BPTT (Tbptt=20), which greatly improves training stability.
Inside an utterance, network states of previous batch are saved and bridged to the next batch as initial history states.

### multi-stream training
multiple utterances are processed simultaneously(4 utterances per CPU in Google's setup).
I prefer to call this "multi-stream". This greatly speeds up the training.
Another reason to do this "multi-stream" training is that all RNN training algorithms are sequential, 
particularly in epoch-wise BPTT, shuffling can only be done in utterance level, whereas frame-level stochasticity is missing.
Multi-stream training receives updates from different utterances at the same time, which improves stochasticity(we are actually using SGD), 

P.S. due to the complexity of multi-stream training, many nnet1 codes are modified, including: cudamatrix & kaldi-matrix kernals, masked loss eval functions etc.
For now i don't recommend to build this version yourself before I make this version "self-contained".  Try simple version first.

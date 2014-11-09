# google's multi-stream version
For architecture details, read the Google paper in this directory.

notable modification to standard LSTM training includes:
### batched-BPTT
As a special case to williams' BPTT(h,h-prime), Google uses a "batched" BPTT (Tbptt=20), this avoids the BPTT time-expension from being too deep, the gradients are less likely to blow up, hence greatly improves training stability in large data set.
Inside an utterance, network states of previous batch are saved as history states and bridged into the next batch.

### multi-stream training
multiple utterances are processed simultaneously(4 utterances per CPU in Google's setup).
I prefer to call this "multi-stream": 
* greatly speeds up the training.
* add sample-level stochasticity, which is benefical in SGD.
To make the second point more clear: epoch-wise BPTT shuffles the training data in utterance level, whereas frame-level stochasticity is missing due to the sequential nature of all RNN trainign algorithms. However, multi-stream training receives updates from different utterances at the same time, which improves stochasticity.

P.S. modifications are made to multiple kaldi source codes, including: cudamatrix & kaldi-matrix kernals, loss functions etc.

### TODO:
* clean up this version, make it self-contained and easy to compile with kaldi codes
* discriminative sequential training(MMI, sMBR) support

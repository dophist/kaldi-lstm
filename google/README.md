# google's multi-stream version
For architecture details, read Google papers in this directory.

notable modifications to standard LSTM training includes:
### batched-BPTT
As a special case to williams' BPTT(h,h-prime), Google uses a "batched" BPTT (Tbptt=20), this avoids the BPTT time-expension from being too deep, the gradients are less likely to blow up, hence greatly improves training stability in large data set.
Inside an utterance, network states of previous batch are saved as history states and bridged into the next batch.

### multi-stream training
Multiple utterances are processed simultaneously(4 utterances per CPU in Google's setup). 

I prefer the term "multi-stream" training
* greatly speed up the training.
* add sample-level stochasticity, which is benefical in SGD.

To elaborate the second point: epoch-wise BPTT shuffles the training data in utterance level, whereas frame-level stochasticity is missing due to the sequential nature of all RNN trainign algorithms. However, multi-stream training receives updates from different utterances at the same time, which improves stochasticity.

The core LSTM algorithms are implemented in nnet/bd-nnet-lstm-projected-streams.h

To speed up the training and get rid of unnecessary matrix buffers, two methods are added to Matrix & CuMatrix:
* AddMatDiagVec() 
* AddMatDotMat()

for more details, see codes in matrix/kaldi-matrix.{h.cc} & cudamatrix/cu-matrix.{h,cc}

Due to the complexity of multi-stream training, a new eval method is added:
* Xent::EvalMasked()
for more details, see codes in nnet/nnet-loss.{h.cc}


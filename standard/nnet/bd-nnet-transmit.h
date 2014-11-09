
#ifndef BD_KALDI_NNET_TRANSMIT_COMPONENT_H_
#define BD_KALDI_NNET_TRANSMIT_COMPONENT_H_


#include "nnet/nnet-component.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-rand.h"

#include "itf/options-itf.h"

namespace kaldi {
namespace nnet1 {


class TransmitComponent : public Component {
 public:
  TransmitComponent(int32 dim_in, int32 dim_out):
      Component(dim_in, dim_out)
  { }
  ~TransmitComponent()
  { }

  Component* Copy() const { return new TransmitComponent(*this); }
  ComponentType GetType() const { return kTransmit; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
      out->CopyFromMat(in);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
          const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
      in_diff->CopyFromMat(out_diff);
  }

};

} // namespace nnet1
} // namespace kaldi

#endif

#ifndef KALDI_NNET_TIME_SHIFT_H_
#define KALDI_NNET_TIME_SHIFT_H_

#include "nnet/nnet-component.h"

namespace kaldi {
namespace nnet1 {


class TimeShift : public Component {
  public:
    TimeShift(int32 dim_in, int32 dim_out):
        Component(dim_in, dim_out),
        shift_(0)
    { }
    ~TimeShift()
    { }

    Component* Copy() const { return new TimeShift(*this); }
    ComponentType GetType() const { return kTimeShift; }

    void InitData(std::istream &is) {
        std::string token; 
        while (!is.eof()) {
            ReadToken(is, false, &token); 
            /**/ if (token == "<Shift>") ReadBasicType(is, false, &shift_);
            else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                           << " (Shift)";
            is >> std::ws; // eat-up whitespace
        }
    }
    void ReadData(std::istream &is, bool binary) {
        ExpectToken(is, binary, "<Shift>");
        ReadBasicType(is, binary, &shift_);
    }
    void WriteData(std::ostream &os, bool binary) const {
        WriteToken(os, binary, "<Shift>");
        WriteBasicType(os, binary, shift_);
        os << "\n";
    }

    void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
        int32 src, dst;
        int32 num_frames = out->NumRows();
        for (dst = 0; dst < num_frames; dst++) {
            src = dst + shift_;
            src = (src < 0) ? 0 : src;
            src = (src > num_frames-1) ? (num_frames-1) : src;
            out->Row(dst).CopyFromVec(in.Row(src));
        }
    }

    void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
          const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
        ;  // it is meaningless to backprop error in TimeShift component
    }
  private:
    int32 shift_;
};

} // namespace nnet1
} // namespace kaldi

#endif

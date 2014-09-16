// nnet/bd-nnet-lstm-precise.h

#ifndef BD_KALDI_NNET_LSTM_PRECISE_H_
#define BD_KALDI_NNET_LSTM_PRECISE_H_

#include "nnet/nnet-component.h"
#include "nnet/nnet-various.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet1 {

class LstmPrecise : public UpdatableComponent {
public:
    LstmPrecise(int32 dim_in, int32 dim_out) :
        UpdatableComponent(dim_in, dim_out), 
        ncell_(dim_out),
        dim_in_(dim_in),
        dim_out_(dim_out),

        w_x_g_(dim_in,  ncell_, kSetZero),
        w_x_i_(dim_in,  ncell_, kSetZero), 
        w_x_f_(dim_in,  ncell_, kSetZero), 
        w_x_o_(dim_in,  ncell_, kSetZero),

        w_r_g_(dim_out, ncell_, kSetZero),
        w_r_i_(dim_out, ncell_, kSetZero),
        w_r_f_(dim_out, ncell_, kSetZero),
        w_r_o_(dim_out, ncell_, kSetZero),

        bias_g_(ncell_, kSetZero),
        bias_i_(ncell_, kSetZero),
        bias_f_(ncell_, kSetZero),
        bias_o_(ncell_, kSetZero),

        w_c_i_(ncell_, kSetZero),
        w_c_f_(ncell_, kSetZero),
        w_c_o_(ncell_, kSetZero),

        w_x_g_corr_(dim_in_, ncell_, kSetZero),
        w_x_i_corr_(dim_in_, ncell_, kSetZero),
        w_x_f_corr_(dim_in_, ncell_, kSetZero),
        w_x_o_corr_(dim_in_, ncell_, kSetZero),

        w_r_g_corr_(dim_out_, ncell_, kSetZero),
        w_r_i_corr_(dim_out_, ncell_, kSetZero),
        w_r_f_corr_(dim_out_, ncell_, kSetZero),
        w_r_o_corr_(dim_out_, ncell_, kSetZero),

        bias_g_corr_(ncell_, kSetZero),
        bias_i_corr_(ncell_, kSetZero),
        bias_f_corr_(ncell_, kSetZero),
        bias_o_corr_(ncell_, kSetZero),

        w_c_i_corr_(ncell_, kSetZero),
        w_c_f_corr_(ncell_, kSetZero),
        w_c_o_corr_(ncell_, kSetZero)
    { }

    ~LstmPrecise()
    { }

    Component* Copy() const { return new LstmPrecise(*this); }
    ComponentType GetType() const { return kLstmPrecise; }

    void InitData(std::istream &is) {
        // define options
        float param_stddev = 0.1;
        // parse config
        std::string token;
        while (!is.eof()) {
            ReadToken(is, false, &token); 
            if(token == "<Inverse>") 
                ReadBasicType(is, false, &inverse_);
            else if (token == "<ParamStddev>") 
                ReadBasicType(is, false, &param_stddev);
            else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                           << " (ParamStddev)";
            is >> std::ws; // eat-up whitespace
        }

        // gaussian initialization
        w_x_g_.SetRandn(); w_x_g_.Scale(param_stddev);
        w_x_i_.SetRandn(); w_x_i_.Scale(param_stddev);
        w_x_f_.SetRandn(); w_x_f_.Scale(param_stddev);
        w_x_o_.SetRandn(); w_x_o_.Scale(param_stddev);

        w_r_g_.SetRandn(); w_r_g_.Scale(param_stddev);
        w_r_i_.SetRandn(); w_r_i_.Scale(param_stddev);
        w_r_f_.SetRandn(); w_r_f_.Scale(param_stddev);
        w_r_o_.SetRandn(); w_r_o_.Scale(param_stddev);

        bias_g_.SetRandn(); bias_g_.Scale(param_stddev);
        bias_i_.SetRandn(); bias_i_.Scale(param_stddev);
        bias_f_.SetRandn(); bias_f_.Scale(param_stddev);
        bias_o_.SetRandn(); bias_o_.Scale(param_stddev);

        w_c_i_.SetRandn(); w_c_i_.Scale(param_stddev);
        w_c_f_.SetRandn(); w_c_f_.Scale(param_stddev);
        w_c_o_.SetRandn(); w_c_o_.Scale(param_stddev);
    }

    void ReadData(std::istream &is, bool binary) {
        ReadBasicType(is, binary, &inverse_);

        w_x_g_.Read(is, binary);
        w_x_i_.Read(is, binary);
        w_x_f_.Read(is, binary);
        w_x_o_.Read(is, binary);

        w_r_g_.Read(is, binary);
        w_r_i_.Read(is, binary);
        w_r_f_.Read(is, binary);
        w_r_o_.Read(is, binary);

        bias_g_.Read(is, binary);
        bias_i_.Read(is, binary);
        bias_f_.Read(is, binary);
        bias_o_.Read(is, binary);

        w_c_i_.Read(is, binary);
        w_c_f_.Read(is, binary);
        w_c_o_.Read(is, binary);
    }

    void WriteData(std::ostream &os, bool binary) const {
        WriteBasicType(os, binary, inverse_);

        w_x_g_.Write(os, binary);
        w_x_i_.Write(os, binary);
        w_x_f_.Write(os, binary);
        w_x_o_.Write(os, binary);

        w_r_g_.Write(os, binary);
        w_r_i_.Write(os, binary);
        w_r_f_.Write(os, binary);
        w_r_o_.Write(os, binary);

        bias_g_.Write(os, binary);
        bias_i_.Write(os, binary);
        bias_f_.Write(os, binary);
        bias_o_.Write(os, binary);

        w_c_i_.Write(os, binary);
        w_c_f_.Write(os, binary);
        w_c_o_.Write(os, binary);
    }

    // TODO
    int32 NumParams() const { 
        return 1;
    }

    // TODO
    void GetParams(Vector<BaseFloat>* wei_copy) const {
        wei_copy->Resize(NumParams());
        return;
    }
    std::string Info() const {
        return std::string("\n   ") + 
            "\n  w_x_g_  " + MomentStatistics(w_x_g_) + 
            "\n  w_x_i_  " + MomentStatistics(w_x_i_) +
            "\n  w_x_f_  " + MomentStatistics(w_x_f_) +
            "\n  w_x_o_  " + MomentStatistics(w_x_o_) +

            "\n  w_r_g_  " + MomentStatistics(w_x_g_) +
            "\n  w_r_i_  " + MomentStatistics(w_x_i_) +
            "\n  w_r_f_  " + MomentStatistics(w_x_f_) +
            "\n  w_r_o_  " + MomentStatistics(w_x_o_) +

            "\n  bias_g_  " + MomentStatistics(bias_g_) +
            "\n  bias_i_  " + MomentStatistics(bias_i_) +
            "\n  bias_f_  " + MomentStatistics(bias_f_) +
            "\n  bias_o_  " + MomentStatistics(bias_o_) +

            "\n  w_c_i_  " + MomentStatistics(w_c_i_) +
            "\n  w_c_f_  " + MomentStatistics(w_c_f_) +
            "\n  w_c_o_  " + MomentStatistics(w_c_o_);
    }
  
    std::string InfoGradient() const {
        return std::string("\n   ") + 
            "\n  w_x_g_corr_  " + MomentStatistics(w_x_g_corr_) + 
            "\n  w_x_i_corr_  " + MomentStatistics(w_x_i_corr_) +
            "\n  w_x_f_corr_  " + MomentStatistics(w_x_f_corr_) +
            "\n  w_x_o_corr_  " + MomentStatistics(w_x_o_corr_) +

            "\n  w_r_g_corr_  " + MomentStatistics(w_x_g_corr_) +
            "\n  w_r_i_corr_  " + MomentStatistics(w_x_i_corr_) +
            "\n  w_r_f_corr_  " + MomentStatistics(w_x_f_corr_) +
            "\n  w_r_o_corr_  " + MomentStatistics(w_x_o_corr_) +

            "\n  bias_g_corr_  " + MomentStatistics(bias_g_corr_) +
            "\n  bias_i_corr_  " + MomentStatistics(bias_i_corr_) +
            "\n  bias_f_corr_  " + MomentStatistics(bias_f_corr_) +
            "\n  bias_o_corr_  " + MomentStatistics(bias_o_corr_) +

            "\n  w_c_i_corr_  " + MomentStatistics(w_c_i_corr_) +
            "\n  w_c_f_corr_  " + MomentStatistics(w_c_f_corr_) +
            "\n  w_c_o_corr_  " + MomentStatistics(w_c_o_corr_);
    }

void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    int32 T = in.NumRows();
    // resize & clear propagate buffers
    yg_.Resize(T, ncell_, kSetZero);
    yi_.Resize(T, ncell_, kSetZero);
    yf_.Resize(T, ncell_, kSetZero);
    yo_.Resize(T, ncell_, kSetZero);
    yc_.Resize(T, ncell_, kSetZero);
    yh_.Resize(T, ncell_, kSetZero);

    CuMatrix<BaseFloat> add_buf(1, ncell_, kUndefined);
    for (int t = 0; t < T; t++) {
        CuSubMatrix<BaseFloat> x(in.RowRange(t,1));
        CuSubMatrix<BaseFloat> y(out->RowRange(t,1));

        CuSubMatrix<BaseFloat> yg(yg_.RowRange(t,1));
        CuSubMatrix<BaseFloat> yi(yi_.RowRange(t,1));
        CuSubMatrix<BaseFloat> yf(yf_.RowRange(t,1));
        CuSubMatrix<BaseFloat> yo(yo_.RowRange(t,1));
        CuSubMatrix<BaseFloat> yc(yc_.RowRange(t,1));
        CuSubMatrix<BaseFloat> yh(yh_.RowRange(t,1));

        // i & f
        if (t > 0) {
            // peephole from c(t-1)
            yi.AddMat(1.0, yc_.RowRange(t-1,1)); yi.MulColsVec(w_c_i_);
            yf.AddMat(1.0, yc_.RowRange(t-1,1)); yf.MulColsVec(w_c_f_);
            // recurrent from y(t-1)
            yi.AddMatMat(1.0, out->RowRange(t-1,1), kNoTrans, w_r_i_, kNoTrans, 1.0);
            yf.AddMatMat(1.0, out->RowRange(t-1,1), kNoTrans, w_r_f_, kNoTrans, 1.0);
        }
        yi.AddMatMat(1.0, x, kNoTrans, w_x_i_, kNoTrans, 1.0);
        yf.AddMatMat(1.0, x, kNoTrans, w_x_f_, kNoTrans, 1.0);

        yi.AddVecToRows(1.0, bias_i_); yi.Sigmoid(yi);
        yf.AddVecToRows(1.0, bias_f_); yf.Sigmoid(yf);

        // g squashing
        if (t > 0) {
            yg.AddMatMat(1.0, out->RowRange(t-1,1), kNoTrans, w_r_g_, kNoTrans, 1.0);
        }
        yg.AddMatMat(1.0, x, kNoTrans, w_x_g_, kNoTrans, 1.0);
        yg.AddVecToRows(1.0, bias_g_, 1.0); yg.Tanh(yg);

        // c memory cell CEC
        if (t > 0) {
            yc.AddMat(1.0, yc_.RowRange(t-1,1)); yc.MulElements(yf);
        }
        add_buf.SetZero();
        add_buf.AddMat(1.0, yg); add_buf.MulElements(yi);
        yc.AddMat(1.0, add_buf);

        // h squashing
        yh.AddMat(1.0, yc);
        yh.Tanh(yh);

        // o output gate
        // peephole to output-gate is from c(t), not c(t-1)
        yo.AddMat(1.0, yc); yo.MulColsVec(w_c_o_);  
        if (t > 0) {
            yo.AddMatMat(1.0, out->RowRange(t-1,1), kNoTrans, w_r_o_, kNoTrans, 1.0);
        }
        yo.AddMatMat(1.0, x, kNoTrans, w_x_o_, kNoTrans, 1.0);
        yo.AddVecToRows(1.0, bias_o_);
        yo.Sigmoid(yo);

        // y
        y.AddMat(1.0, yh); y.MulElements(yo);
    }
}

void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    int32 T = in.NumRows();

    diff_g_.Resize(T, ncell_, kSetZero);
    diff_i_.Resize(T, ncell_, kSetZero);
    diff_f_.Resize(T, ncell_, kSetZero);
    diff_o_.Resize(T, ncell_, kSetZero);
    diff_c_.Resize(T, ncell_, kSetZero);
    diff_h_.Resize(T, ncell_, kSetZero);
    diff_m_.Resize(T, ncell_, kSetZero);

    CuMatrix<BaseFloat> add_buf(1, ncell_, kUndefined);

    for (int t = T-1; t >= 0; t--) {

        CuSubMatrix<BaseFloat> yg(yg_.RowRange(t,1));
        CuSubMatrix<BaseFloat> yi(yi_.RowRange(t,1));
        CuSubMatrix<BaseFloat> yf(yf_.RowRange(t,1));
        CuSubMatrix<BaseFloat> yo(yo_.RowRange(t,1));
        CuSubMatrix<BaseFloat> yc(yc_.RowRange(t,1));
        CuSubMatrix<BaseFloat> yh(yh_.RowRange(t,1));

        CuSubMatrix<BaseFloat> diff_g(diff_g_.RowRange(t,1));
        CuSubMatrix<BaseFloat> diff_i(diff_i_.RowRange(t,1));
        CuSubMatrix<BaseFloat> diff_f(diff_f_.RowRange(t,1));
        CuSubMatrix<BaseFloat> diff_o(diff_o_.RowRange(t,1));
        CuSubMatrix<BaseFloat> diff_c(diff_c_.RowRange(t,1));
        CuSubMatrix<BaseFloat> diff_h(diff_h_.RowRange(t,1));
        CuSubMatrix<BaseFloat> diff_m(diff_m_.RowRange(t,1));

        // m
        if (t < T-1) {
            diff_m.AddMatMat(1.0, diff_g_.RowRange(t+1,1), kNoTrans, w_r_g_, kTrans, 1.0);
            diff_m.AddMatMat(1.0, diff_i_.RowRange(t+1,1), kNoTrans, w_r_i_, kTrans, 1.0);  //
            diff_m.AddMatMat(1.0, diff_f_.RowRange(t+1,1), kNoTrans, w_r_f_, kTrans, 1.0);  // |-> commment out these 3 lines, the equations becomes the same as PhD disseration of Alex Graves
            diff_m.AddMatMat(1.0, diff_o_.RowRange(t+1,1), kNoTrans, w_r_o_, kTrans, 1.0);  //
        }
        diff_m.AddMat(1.0, out_diff.RowRange(t,1));

        // h
        diff_h.AddMat(1.0, diff_m); diff_h.MulElements(yo);
        diff_h.DiffTanh(yh, diff_h);

        // o
        diff_o.AddMat(1.0, diff_m); diff_o.MulElements(yh);
        diff_o.DiffSigmoid(yo, diff_o);

        // c
        //     1. diff from h(t)
        diff_c.AddMat(1.0, diff_h);  
        //     2. diff from output-gate(t) (via peephole)
        add_buf.SetZero();    
        add_buf.AddMat(1.0, diff_o); add_buf.MulColsVec(w_c_o_);  
        diff_c.AddMat(1.0, add_buf);  

        if (t < T-1) {
            // 3. diff from c(t+1) (via forget-gate between CEC)
            add_buf.SetZero();    
            add_buf.AddMat(1.0, diff_c_.RowRange(t+1,1)); add_buf.MulElements(yf_.RowRange(t+1,1));  
            diff_c.AddMat(1.0, add_buf);  
            // 4. diff from forget-gate(t+1) (via peephole)
            add_buf.SetZero();  
            add_buf.AddMat(1.0, diff_f_.RowRange(t+1,1)); add_buf.MulColsVec(w_c_f_);  
            diff_c.AddMat(1.0, add_buf);
            // 5. diff from input-gate(t+1) (via peephole)
            add_buf.SetZero();    
            add_buf.AddMat(1.0, diff_i_.RowRange(t+1,1)); add_buf.MulColsVec(w_c_i_);  
            diff_c.AddMat(1.0, add_buf);
        }

        // f
        if (t > 0) {
            diff_f.AddMat(1.0, yc_.RowRange(t-1,1));
            diff_f.MulElements(diff_c);
            diff_f.DiffSigmoid(yf, diff_f);
        }

        // i
        diff_i.AddMat(1.0, diff_c);
        diff_i.MulElements(yg);
        diff_i.DiffSigmoid(yi, diff_i);

        // g
        diff_g.AddMat(1.0, diff_c);
        diff_g.MulElements(yi);
        diff_g.DiffTanh(yg, diff_g);

        // x
        in_diff->RowRange(t,1).AddMatMat(1.0, diff_g, kNoTrans, w_x_g_, kTrans, 1.0);
        in_diff->RowRange(t,1).AddMatMat(1.0, diff_i, kNoTrans, w_x_i_, kTrans, 1.0);
        in_diff->RowRange(t,1).AddMatMat(1.0, diff_f, kNoTrans, w_x_f_, kTrans, 1.0);
        in_diff->RowRange(t,1).AddMatMat(1.0, diff_o, kNoTrans, w_x_o_, kTrans, 1.0);
    }

    // calculate delta
    const BaseFloat lr = opts_.learn_rate;
    const BaseFloat mmt = opts_.momentum;

    w_x_g_corr_.AddMatMat(-lr, in, kTrans, diff_g_, kNoTrans, mmt);
    w_x_i_corr_.AddMatMat(-lr, in, kTrans, diff_i_, kNoTrans, mmt);
    w_x_f_corr_.AddMatMat(-lr, in, kTrans, diff_f_, kNoTrans, mmt);
    w_x_o_corr_.AddMatMat(-lr, in, kTrans, diff_o_, kNoTrans, mmt);

    w_r_g_corr_.AddMatMat(-lr, out.RowRange(0,T-1), kTrans, diff_g_.RowRange(1,T-1), kNoTrans, mmt);
    w_r_i_corr_.AddMatMat(-lr, out.RowRange(0,T-1), kTrans, diff_i_.RowRange(1,T-1), kNoTrans, mmt);
    w_r_f_corr_.AddMatMat(-lr, out.RowRange(0,T-1), kTrans, diff_f_.RowRange(1,T-1), kNoTrans, mmt);
    w_r_o_corr_.AddMatMat(-lr, out.RowRange(0,T-1), kTrans, diff_o_.RowRange(1,T-1), kNoTrans, mmt);

    bias_g_corr_.AddRowSumMat(-lr, diff_g_, mmt);
    bias_i_corr_.AddRowSumMat(-lr, diff_i_, mmt);
    bias_f_corr_.AddRowSumMat(-lr, diff_f_, mmt);
    bias_o_corr_.AddRowSumMat(-lr, diff_o_, mmt);

    CuMatrix<BaseFloat> buf(T, ncell_, kUndefined);

    buf.SetZero();
    buf.RowRange(0,T-1).CopyFromMat(yc_.RowRange(0,T-1));
    buf.RowRange(0,T-1).MulElements(diff_i_.RowRange(1, T-1));
    w_c_i_corr_.AddRowSumMat(-lr, buf.RowRange(0,T-1), mmt);

    buf.SetZero();
    buf.RowRange(0,T-1).CopyFromMat(yc_.RowRange(0,T-1));
    buf.RowRange(0,T-1).MulElements(diff_f_.RowRange(1, T-1));
    w_c_f_corr_.AddRowSumMat(-lr, buf.RowRange(0,T-1), mmt);

    buf.SetZero();
    buf.CopyFromMat(yc_);
    buf.MulElements(diff_o_);
    w_c_o_corr_.AddRowSumMat(-lr, buf, mmt);
}

void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
    w_x_g_.AddMat(1.0, w_x_g_corr_);
    w_x_i_.AddMat(1.0, w_x_i_corr_);
    w_x_f_.AddMat(1.0, w_x_f_corr_);
    w_x_o_.AddMat(1.0, w_x_o_corr_);

    w_r_g_.AddMat(1.0, w_r_g_corr_);
    w_r_i_.AddMat(1.0, w_r_i_corr_);
    w_r_f_.AddMat(1.0, w_r_f_corr_);
    w_r_o_.AddMat(1.0, w_r_o_corr_);

    bias_g_.AddVec(1.0, bias_g_corr_);
    bias_i_.AddVec(1.0, bias_i_corr_);
    bias_f_.AddVec(1.0, bias_f_corr_);
    bias_o_.AddVec(1.0, bias_o_corr_);

    w_c_i_.AddVec(1.0, w_c_i_corr_);
    w_c_f_.AddVec(1.0, w_c_f_corr_);
    w_c_o_.AddVec(1.0, w_c_o_corr_);
}

private:
    /*************************************
     * x: input neuron
     * g: squashing neuron near input
     * i: Input gate
     * f: Forget gate
     * o: Output gate
     * c: memory Cell (cec)
     * h: squashing neuron near output
     * m: output neuron of Memory block
     *************************************/
    /* network parameters */
    int32 ncell_;
    int32 dim_in_;
    int32 dim_out_;
    bool inverse_;

    // forward connection
    CuMatrix<BaseFloat> w_x_g_;
    CuMatrix<BaseFloat> w_x_i_;
    CuMatrix<BaseFloat> w_x_f_;
    CuMatrix<BaseFloat> w_x_o_;

    // recursive connection
    CuMatrix<BaseFloat> w_r_g_;
    CuMatrix<BaseFloat> w_r_i_;
    CuMatrix<BaseFloat> w_r_f_;
    CuMatrix<BaseFloat> w_r_o_;

    // bias
    CuVector<BaseFloat> bias_g_;
    CuVector<BaseFloat> bias_i_;
    CuVector<BaseFloat> bias_f_;
    CuVector<BaseFloat> bias_o_;

    // peephole connection
    CuVector<BaseFloat> w_c_i_;
    CuVector<BaseFloat> w_c_f_;
    CuVector<BaseFloat> w_c_o_;

    /* propagate buffers */
    CuMatrix<BaseFloat> yg_;
    CuMatrix<BaseFloat> yi_;
    CuMatrix<BaseFloat> yf_;
    CuMatrix<BaseFloat> yo_;
    CuMatrix<BaseFloat> yc_;
    CuMatrix<BaseFloat> yh_;

    /* back-propagate buffers */
    CuMatrix<BaseFloat> diff_g_;
    CuMatrix<BaseFloat> diff_i_;
    CuMatrix<BaseFloat> diff_f_;
    CuMatrix<BaseFloat> diff_o_;
    CuMatrix<BaseFloat> diff_c_;
    CuMatrix<BaseFloat> diff_h_;
    CuMatrix<BaseFloat> diff_m_;

    /* delta buffers */
    CuMatrix<BaseFloat> w_x_g_corr_;
    CuMatrix<BaseFloat> w_x_i_corr_;
    CuMatrix<BaseFloat> w_x_f_corr_;
    CuMatrix<BaseFloat> w_x_o_corr_;

    CuMatrix<BaseFloat> w_r_g_corr_;
    CuMatrix<BaseFloat> w_r_i_corr_;
    CuMatrix<BaseFloat> w_r_f_corr_;
    CuMatrix<BaseFloat> w_r_o_corr_;

    CuVector<BaseFloat> bias_g_corr_;
    CuVector<BaseFloat> bias_i_corr_;
    CuVector<BaseFloat> bias_f_corr_;
    CuVector<BaseFloat> bias_o_corr_;

    CuVector<BaseFloat> w_c_i_corr_;
    CuVector<BaseFloat> w_c_f_corr_;
    CuVector<BaseFloat> w_c_o_corr_;
};

} // namespace nnet1
} // namespace kaldi

#endif

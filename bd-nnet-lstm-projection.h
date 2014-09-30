// nnet/bd-nnet-lstm-projection.h

#ifndef BD_KALDI_NNET_LSTM_PROJECTION_H_
#define BD_KALDI_NNET_LSTM_PROJECTION_H_

#include "nnet/nnet-component.h"
#include "nnet/nnet-various.h"
#include "cudamatrix/cu-math.h"

/*************************************
 * x: input neuron
 * g: squashing neuron near input
 * i: Input gate
 * f: Forget gate
 * o: Output gate
 * c: memory Cell (CEC)
 * h: squashing neuron near output
 * m: output neuron of Memory block
 * r: recurrent projection neuron
 *************************************/

namespace kaldi {
namespace nnet1 {
class LstmProjection : public UpdatableComponent {
public:
    LstmProjection(int32 input_dim, int32 output_dim) :
        UpdatableComponent(input_dim, output_dim),
        ncell_(0),
        nrecur_(output_dim)  // recurrent projection layer is also feed-forward as LSTM output
    { }

    ~LstmProjection()
    { }

    Component* Copy() const { return new LstmProjection(*this); }
    ComponentType GetType() const { return kLstmProjection; }

    static void InitMatParam(CuMatrix<BaseFloat> &m, float scale) {
        m.SetRandUniform();  // uniform in [0, 1]
        m.Add(-0.5);         // uniform in [-0.5, 0.5]
        m.Scale(2 * scale);  // uniform in [-scale, +scale]
    }

    static void InitVecParam(CuVector<BaseFloat> &v, float scale) {
        Vector<BaseFloat> tmp(v.Dim());
        for (int i=0; i < tmp.Dim(); i++) {
            tmp(i) = (RandUniform() - 0.5) * 2 * scale;
        }
        v = tmp;
    }

    void InitData(std::istream &is) {
        // define options
        float param_scale = 0.02;
        // parse config
        std::string token;
        while (!is.eof()) {
            ReadToken(is, false, &token); 
            if (token == "<CellDim>") 
                ReadBasicType(is, false, &ncell_);
            else if (token == "<ParamScale>") 
                ReadBasicType(is, false, &param_scale);
            else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                           << " (CellDim|ParamScale)";
            is >> std::ws; // eat-up whitespace
        }

//        // init weight and bias (Gaussian)
//        w_gifo_x_.Resize(4*ncell_, input_dim_); 
//        w_gifo_x_.SetRandn();  w_gifo_x_.Scale(param_scale);
//
//        w_gifo_r_.Resize(4*ncell_, nrecur_);    
//        w_gifo_r_.SetRandn();  w_gifo_r_.Scale(param_scale);
//
//        bias_.Resize(4*ncell_);     
//        bias_.SetRandn();  bias_.Scale(param_scale);
//
//        peephole_i_c_.Resize(ncell_);
//        peephole_i_c_.SetRandn();  peephole_i_c_.Scale(param_scale);
//
//        peephole_f_c_.Resize(ncell_);
//        peephole_f_c_.SetRandn();  peephole_f_c_.Scale(param_scale);
//
//        peephole_o_c_.Resize(ncell_);
//        peephole_o_c_.SetRandn();  peephole_o_c_.Scale(param_scale);
//
//        w_r_m_.Resize(nrecur_, ncell_); 
//        w_r_m_.SetRandn();  w_r_m_.Scale(param_scale);

        // init weight and bias (Uniform)
        w_gifo_x_.Resize(4*ncell_, input_dim_);  InitMatParam(w_gifo_x_, param_scale);
        w_gifo_r_.Resize(4*ncell_, nrecur_);     InitMatParam(w_gifo_r_, param_scale);
        w_r_m_.Resize(nrecur_, ncell_);          InitMatParam(w_r_m_, param_scale);

        bias_.Resize(4*ncell_);        InitVecParam(bias_, param_scale);
        peephole_i_c_.Resize(ncell_);  InitVecParam(peephole_i_c_, param_scale);
        peephole_f_c_.Resize(ncell_);  InitVecParam(peephole_f_c_, param_scale);
        peephole_o_c_.Resize(ncell_);  InitVecParam(peephole_o_c_, param_scale);

        // init delta buffers
        w_gifo_x_corr_.Resize(4*ncell_, input_dim_, kSetZero); 
        w_gifo_r_corr_.Resize(4*ncell_, nrecur_, kSetZero);    
        bias_corr_.Resize(4*ncell_, kSetZero);     

        peephole_i_c_corr_.Resize(ncell_, kSetZero);
        peephole_f_c_corr_.Resize(ncell_, kSetZero);
        peephole_o_c_corr_.Resize(ncell_, kSetZero);

        w_r_m_corr_.Resize(nrecur_, ncell_, kSetZero); 
    }

    void ReadData(std::istream &is, bool binary) {
        ExpectToken(is, binary, "<CellDim>");
        ReadBasicType(is, binary, &ncell_);

        w_gifo_x_.Read(is, binary);
        w_gifo_r_.Read(is, binary);
        bias_.Read(is, binary);

        peephole_i_c_.Read(is, binary);
        peephole_f_c_.Read(is, binary);
        peephole_o_c_.Read(is, binary);

        w_r_m_.Read(is, binary);

        // init delta buffers
        w_gifo_x_corr_.Resize(4*ncell_, input_dim_, kSetZero); 
        w_gifo_r_corr_.Resize(4*ncell_, nrecur_, kSetZero);    
        bias_corr_.Resize(4*ncell_, kSetZero);     

        peephole_i_c_corr_.Resize(ncell_, kSetZero);
        peephole_f_c_corr_.Resize(ncell_, kSetZero);
        peephole_o_c_corr_.Resize(ncell_, kSetZero);

        w_r_m_corr_.Resize(nrecur_, ncell_, kSetZero); 
    }

    void WriteData(std::ostream &os, bool binary) const {
        WriteToken(os, binary, "<CellDim>");
        WriteBasicType(os, binary, ncell_);

        w_gifo_x_.Write(os, binary);
        w_gifo_r_.Write(os, binary);
        bias_.Write(os, binary);

        peephole_i_c_.Write(os, binary);
        peephole_f_c_.Write(os, binary);
        peephole_o_c_.Write(os, binary);

        w_r_m_.Write(os, binary);
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
        return std::string("    ") + 
            "\n  w_gifo_x_  "     + MomentStatistics(w_gifo_x_) + 
            "\n  w_gifo_r_  "     + MomentStatistics(w_gifo_r_) +
            "\n  bias_  "         + MomentStatistics(bias_) +
            "\n  peephole_i_c_  " + MomentStatistics(peephole_i_c_) +
            "\n  peephole_f_c_  " + MomentStatistics(peephole_f_c_) +
            "\n  peephole_o_c_  " + MomentStatistics(peephole_o_c_) +
            "\n  w_r_m_  "        + MomentStatistics(w_r_m_);
    }
  
    std::string InfoGradient() const {
        return std::string("    ") + 
            "\n  w_gifo_x_corr_  "     + MomentStatistics(w_gifo_x_corr_) + 
            "\n  w_gifo_r_corr_  "     + MomentStatistics(w_gifo_r_corr_) +
            "\n  bias_corr_  "         + MomentStatistics(bias_corr_) +
            "\n  peephole_i_c_corr_  " + MomentStatistics(peephole_i_c_corr_) +
            "\n  peephole_f_c_corr_  " + MomentStatistics(peephole_f_c_corr_) +
            "\n  peephole_o_c_corr_  " + MomentStatistics(peephole_o_c_corr_) +
            "\n  w_r_m_corr_  "        + MomentStatistics(w_r_m_corr_);
    }

    void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
        int DEBUG = 0;
        int32 T = in.NumRows();
        // resize & clear propagate buffers
        propagate_buf_.Resize(T, 7 * ncell_ + nrecur_, kSetZero);

        // disassemble entire neuron activation buffer into different neurons
        CuSubMatrix<BaseFloat> YG(propagate_buf_.ColRange(0*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YI(propagate_buf_.ColRange(1*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YF(propagate_buf_.ColRange(2*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YO(propagate_buf_.ColRange(3*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YC(propagate_buf_.ColRange(4*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YH(propagate_buf_.ColRange(5*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YM(propagate_buf_.ColRange(6*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YR(propagate_buf_.ColRange(7*ncell_, nrecur_));

        CuSubMatrix<BaseFloat> YGIFO(propagate_buf_.ColRange(0, 4*ncell_));

        // (x & bias) -> g, i, f, o, not recursive, do it all in once
        YGIFO.AddMatMat(1.0, in, kNoTrans, w_gifo_x_, kTrans, 0.0);
        YGIFO.AddVecToRows(1.0, bias_);

        for (int t = 0; t < T; t++) {
            // (vector & matrix) representations of neuron activations at frame t 
            // so we can borrow rich APIs from both CuMatrix and CuVector
            CuSubVector<BaseFloat> y_g(YG.Row(t));  CuSubMatrix<BaseFloat> YG_t(YG.RowRange(t,1));  
            CuSubVector<BaseFloat> y_i(YI.Row(t));  CuSubMatrix<BaseFloat> YI_t(YI.RowRange(t,1));  
            CuSubVector<BaseFloat> y_f(YF.Row(t));  CuSubMatrix<BaseFloat> YF_t(YF.RowRange(t,1));  
            CuSubVector<BaseFloat> y_o(YO.Row(t));  CuSubMatrix<BaseFloat> YO_t(YO.RowRange(t,1));  
            CuSubVector<BaseFloat> y_c(YC.Row(t));  CuSubMatrix<BaseFloat> YC_t(YC.RowRange(t,1));  
            CuSubVector<BaseFloat> y_h(YH.Row(t));  CuSubMatrix<BaseFloat> YH_t(YH.RowRange(t,1));  
            CuSubVector<BaseFloat> y_m(YM.Row(t));  CuSubMatrix<BaseFloat> YM_t(YM.RowRange(t,1));  
            CuSubVector<BaseFloat> y_r(YR.Row(t));  CuSubMatrix<BaseFloat> YR_t(YR.RowRange(t,1));  

            CuSubVector<BaseFloat> y_gifo(YGIFO.Row(t));
    
            if (t > 0) {
                // recursion r(t-1) -> g, i, f, o
                y_gifo.AddMatVec(1.0, w_gifo_r_, kNoTrans, YR.Row(t-1), 1.0);
                // peephole c(t-1) -> i(t)
                y_i.AddVecVec(1.0, peephole_i_c_, YC.Row(t-1), 1.0);
                // peephole c(t-1) -> f(t)
                y_f.AddVecVec(1.0, peephole_f_c_, YC.Row(t-1), 1.0);
            }

            // i, f sigmoid squashing
            YI_t.Sigmoid(YI_t);
            YF_t.Sigmoid(YF_t);
    
            // g tanh squashing
            YG_t.Tanh(YG_t);
    
            // c memory cell
            y_c.AddVecVec(1.0, y_i, y_g, 0.0);
            if (t > 0) {  // CEC connection via forget gate: c(t-1) -> c(t)
                y_c.AddVecVec(1.0, y_f, YC.Row(t-1), 1.0);
            }

            YC_t.ApplyFloor(-50);   // optional clipping of cell activation
            YC_t.ApplyCeiling(50);  // google paper Interspeech2014: LSTM for LVCSR
    
            // h tanh squashing
            YH_t.Tanh(YC_t);
    
            // o output gate
            y_o.AddVecVec(1.0, peephole_o_c_, y_c, 1.0);  // notice: output gate peephole is not recursive
            YO_t.Sigmoid(YO_t);
    
            // m
            y_m.AddVecVec(1.0, y_o, y_h, 0.0);
            
            // r
            y_r.AddMatVec(1.0, w_r_m_, kNoTrans, y_m, 0.0);

            if (DEBUG) {
                std::cerr << "propagate frame " << t << "\n";
                std::cerr << "activation of g: " << y_g;
                std::cerr << "activation of i: " << y_i;
                std::cerr << "activation of f: " << y_f;
                std::cerr << "activation of o: " << y_o;
                std::cerr << "activation of c: " << y_c;
                std::cerr << "activation of h: " << y_h;
                std::cerr << "activation of m: " << y_m;
                std::cerr << "activation of r: " << y_r;
            }
        }

        // recurrent projection layer is also feed-forward as LSTM output
        out->CopyFromMat(YR);
    }

    void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                            const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
        int DEBUG = 0;
        int32 T = in.NumRows();
        // disassemble propagated buffer into neurons
        CuSubMatrix<BaseFloat> YG(propagate_buf_.ColRange(0*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YI(propagate_buf_.ColRange(1*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YF(propagate_buf_.ColRange(2*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YO(propagate_buf_.ColRange(3*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YC(propagate_buf_.ColRange(4*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YH(propagate_buf_.ColRange(5*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YM(propagate_buf_.ColRange(6*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YR(propagate_buf_.ColRange(7*ncell_, nrecur_));
    
        // 0-init backpropagate buffer
        backpropagate_buf_.Resize(T, 7 * ncell_ + nrecur_, kSetZero);

        // disassemble backpropagate buffer into neurons
        CuSubMatrix<BaseFloat> DG(backpropagate_buf_.ColRange(0*ncell_, ncell_));
        CuSubMatrix<BaseFloat> DI(backpropagate_buf_.ColRange(1*ncell_, ncell_));
        CuSubMatrix<BaseFloat> DF(backpropagate_buf_.ColRange(2*ncell_, ncell_));
        CuSubMatrix<BaseFloat> DO(backpropagate_buf_.ColRange(3*ncell_, ncell_));
        CuSubMatrix<BaseFloat> DC(backpropagate_buf_.ColRange(4*ncell_, ncell_));
        CuSubMatrix<BaseFloat> DH(backpropagate_buf_.ColRange(5*ncell_, ncell_));
        CuSubMatrix<BaseFloat> DM(backpropagate_buf_.ColRange(6*ncell_, ncell_));
        CuSubMatrix<BaseFloat> DR(backpropagate_buf_.ColRange(7*ncell_, nrecur_));

        CuSubMatrix<BaseFloat> DGIFO(backpropagate_buf_.ColRange(0, 4*ncell_));

        // projection layer to LSTM output is not recursive, so backprop it all in once
        DR.CopyFromMat(out_diff);

        for (int t = T-1; t >= 0; t--) {
            // vector representation                  // matrix representation
            CuSubVector<BaseFloat> y_g(YG.Row(t));    CuSubMatrix<BaseFloat> YG_t(YG.RowRange(t,1));  
            CuSubVector<BaseFloat> y_i(YI.Row(t));    CuSubMatrix<BaseFloat> YI_t(YI.RowRange(t,1));  
            CuSubVector<BaseFloat> y_f(YF.Row(t));    CuSubMatrix<BaseFloat> YF_t(YF.RowRange(t,1));  
            CuSubVector<BaseFloat> y_o(YO.Row(t));    CuSubMatrix<BaseFloat> YO_t(YO.RowRange(t,1));  
            CuSubVector<BaseFloat> y_c(YC.Row(t));    CuSubMatrix<BaseFloat> YC_t(YC.RowRange(t,1));  
            CuSubVector<BaseFloat> y_h(YH.Row(t));    CuSubMatrix<BaseFloat> YH_t(YH.RowRange(t,1));  
            CuSubVector<BaseFloat> y_m(YM.Row(t));    CuSubMatrix<BaseFloat> YM_t(YM.RowRange(t,1));  
            CuSubVector<BaseFloat> y_r(YR.Row(t));    CuSubMatrix<BaseFloat> YR_t(YR.RowRange(t,1));  
    
            CuSubVector<BaseFloat> d_g(DG.Row(t));    CuSubMatrix<BaseFloat> DG_t(DG.RowRange(t,1));
            CuSubVector<BaseFloat> d_i(DI.Row(t));    CuSubMatrix<BaseFloat> DI_t(DI.RowRange(t,1));
            CuSubVector<BaseFloat> d_f(DF.Row(t));    CuSubMatrix<BaseFloat> DF_t(DF.RowRange(t,1));
            CuSubVector<BaseFloat> d_o(DO.Row(t));    CuSubMatrix<BaseFloat> DO_t(DO.RowRange(t,1));
            CuSubVector<BaseFloat> d_c(DC.Row(t));    CuSubMatrix<BaseFloat> DC_t(DC.RowRange(t,1));
            CuSubVector<BaseFloat> d_h(DH.Row(t));    CuSubMatrix<BaseFloat> DH_t(DH.RowRange(t,1));
            CuSubVector<BaseFloat> d_m(DM.Row(t));    CuSubMatrix<BaseFloat> DM_t(DM.RowRange(t,1));
            CuSubVector<BaseFloat> d_r(DR.Row(t));    CuSubMatrix<BaseFloat> DR_t(DR.RowRange(t,1));
    
            if (t < T-1) {
                // Alex Graves didn't backpropagate error from i(t+1), f(t+1), o(t+1) to r(t) 
                // in his PhD dissertation, however, we compute *precise* gradient here
                d_r.AddMatVec(1.0, w_gifo_r_, kTrans, DGIFO.Row(t+1), 1.0);
            }
    
            // m
            d_m.AddMatVec(1.0, w_r_m_, kTrans, d_r, 0.0);
    
            // h
            d_h.AddVecVec(1.0, y_o, d_m, 0.0);
            DH_t.DiffTanh(YH_t, DH_t);
    
            // o
            d_o.AddVecVec(1.0, y_h, d_m, 1.0);
            DO_t.DiffSigmoid(YO_t, DO_t);
    
            // c
            //     1. diff from h(t)
            d_c.AddVec(1.0, d_h, 0.0);  
            //     2. diff from output-gate(t) (via peephole)
            d_c.AddVecVec(1.0, peephole_o_c_, d_o, 1.0);
    
            if (t < T-1) {
                // 3. diff from c(t+1) (via forget-gate between CEC)
                d_c.AddVecVec(1.0, YF.Row(t+1), DC.Row(t+1), 1.0);
                // 4. diff from forget-gate(t+1) (via peephole)
                d_c.AddVecVec(1.0, peephole_f_c_ , DF.Row(t+1), 1.0);
                // 5. diff from input-gate(t+1) (via peephole)
                d_c.AddVecVec(1.0, peephole_i_c_, DI.Row(t+1), 1.0);
            }
    
            // f
            if (t > 0) {
                d_f.AddVecVec(1.0, YC.Row(t-1), d_c, 0.0);
                DF_t.DiffSigmoid(YF_t, DF_t);
            }
    
            // i
            d_i.AddVecVec(1.0, y_g, d_c, 0.0);
            DI_t.DiffSigmoid(YI_t, DI_t);
    
            // g
            d_g.AddVecVec(1.0, y_i, d_c, 0.0);
            DG_t.DiffTanh(YG_t, DG_t);
    
            // debug info
            if (DEBUG) {
                std::cerr << "backprop frame " << t << "\n";
                std::cerr << "derivative of input y " << out_diff.RowRange(t,1);
                std::cerr << "derivative of input r " << d_r;
                std::cerr << "derivative of input m " << d_m;
                std::cerr << "derivative of input h " << d_h;
                std::cerr << "derivative of input o " << d_o;
                std::cerr << "derivative of input c " << d_c;
                std::cerr << "derivative of input f " << d_f;
                std::cerr << "derivative of input i " << d_i;
                std::cerr << "derivative of input g " << d_g;
            }
        }

        // backprop derivatives to input x, do it all in once
        in_diff->AddMatMat(1.0, DGIFO, kNoTrans, w_gifo_x_, kNoTrans, 0.0);
    
        // calculate delta
        const BaseFloat lr = opts_.learn_rate;
        const BaseFloat mmt = opts_.momentum;
    
        w_gifo_x_corr_.AddMatMat(-lr, DGIFO, kTrans, in, kNoTrans, mmt);
        w_gifo_r_corr_.AddMatMat(-lr, DGIFO.RowRange(1,T-1), kTrans, YR.RowRange(0,T-1), kNoTrans, mmt);

        bias_corr_.AddRowSumMat(-lr, DGIFO, mmt);
    
        peephole_i_c_corr_.AddDiagMatMat(-lr, DI.RowRange(1,T-1), kTrans, YC.RowRange(0,T-1), kNoTrans, mmt);
        peephole_f_c_corr_.AddDiagMatMat(-lr, DF.RowRange(1,T-1), kTrans, YC.RowRange(0,T-1), kNoTrans, mmt);
        peephole_o_c_corr_.AddDiagMatMat(-lr, DO                , kTrans, YC                , kNoTrans, mmt);

        w_r_m_corr_.AddMatMat(-lr, DR, kTrans, YM, kNoTrans, mmt);
    
        if (DEBUG) {
            std::cerr << "update delta: \n";
            std::cerr << "w_gifo_x_corr_ " << w_gifo_x_corr_;
            std::cerr << "w_gifo_r_corr_ " << w_gifo_r_corr_;
            std::cerr << "bias_corr_ " << bias_corr_;
            std::cerr << "w_r_m_corr_ " << w_r_m_corr_;
            std::cerr << "peephole_i_c_corr_ " << peephole_i_c_corr_;
            std::cerr << "peephole_f_c_corr_ " << peephole_f_c_corr_;
            std::cerr << "peephole_o_c_corr_ " << peephole_o_c_corr_;
        }
    }

    void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
        w_gifo_x_.AddMat(1.0, w_gifo_x_corr_);
        w_gifo_r_.AddMat(1.0, w_gifo_r_corr_);
        bias_.AddVec(1.0, bias_corr_, 1.0);
    
        peephole_i_c_.AddVec(1.0, peephole_i_c_corr_, 1.0);
        peephole_f_c_.AddVec(1.0, peephole_f_c_corr_, 1.0);
        peephole_o_c_.AddVec(1.0, peephole_o_c_corr_, 1.0);
    
        w_r_m_.AddMat(1.0, w_r_m_corr_);
    }

private:
    // network topology parameters
    int32 ncell_;   // number of cells (one cell per memory block)
    int32 nrecur_;  // size of recurrent projection layer

    // feed-forward connections: from x to [g, i, f, o]
    CuMatrix<BaseFloat> w_gifo_x_;
    CuMatrix<BaseFloat> w_gifo_x_corr_;

    // recurrent projection connections: from r to [g, i, f, o]
    CuMatrix<BaseFloat> w_gifo_r_;
    CuMatrix<BaseFloat> w_gifo_r_corr_;

    // biases of [g, i, f, o]
    CuVector<BaseFloat> bias_;
    CuVector<BaseFloat> bias_corr_;

    // peephole from c to i, f, g 
    // peephole connections are block-internal, so we use vector form
    CuVector<BaseFloat> peephole_i_c_;
    CuVector<BaseFloat> peephole_f_c_;
    CuVector<BaseFloat> peephole_o_c_;

    CuVector<BaseFloat> peephole_i_c_corr_;
    CuVector<BaseFloat> peephole_f_c_corr_;
    CuVector<BaseFloat> peephole_o_c_corr_;

    // projection layer r: from m to r
    CuMatrix<BaseFloat> w_r_m_;
    CuMatrix<BaseFloat> w_r_m_corr_;

    // propagate buffer: output of [g, i, f, o, c, h, m, r]
    CuMatrix<BaseFloat> propagate_buf_;

    // back-propagate buffer: diff-input of [g, i, f, o, c, h, m, r]
    CuMatrix<BaseFloat> backpropagate_buf_;

};
} // namespace nnet1
} // namespace kaldi

#endif

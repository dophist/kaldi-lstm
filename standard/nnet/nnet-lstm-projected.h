// nnet/nnet-lstm-projected.h
// nnet/nnet-affine-transform.h

// Copyright 2014 author: Jiayu DU, Wei LI @ Baidu Inc.

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_NNET_LSTM_PROJECTED_H_
#define KALDI_NNET_LSTM_PROJECTED_H_

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
 * y: output neuron of LSTMP
 *************************************/

namespace kaldi {
namespace nnet1 {
class LstmProjected : public UpdatableComponent {
public:
    LstmProjected(int32 input_dim, int32 output_dim) :
        UpdatableComponent(input_dim, output_dim),
        ncell_(0),
        nrecur_(output_dim)
    { }

    ~LstmProjected()
    { }

    Component* Copy() const { return new LstmProjected(*this); }
    ComponentType GetType() const { return kLstmProjected; }

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

        //prev_nnet_state_.Resize(7*ncell_ + 1*nrecur_, kSetZero);

        // init weight and bias (Uniform)
        w_gifo_x_.Resize(4*ncell_, input_dim_, kUndefined);  InitMatParam(w_gifo_x_, param_scale);
        w_gifo_r_.Resize(4*ncell_, nrecur_, kUndefined);     InitMatParam(w_gifo_r_, param_scale);
        w_r_m_.Resize(nrecur_, ncell_, kUndefined);          InitMatParam(w_r_m_, param_scale);

        bias_.Resize(4*ncell_, kUndefined);        InitVecParam(bias_, param_scale);
        peephole_i_c_.Resize(ncell_, kUndefined);  InitVecParam(peephole_i_c_, param_scale);
        peephole_f_c_.Resize(ncell_, kUndefined);  InitVecParam(peephole_f_c_, param_scale);
        peephole_o_c_.Resize(ncell_, kUndefined);  InitVecParam(peephole_o_c_, param_scale);

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

        //prev_nnet_state_.Resize(7*ncell_ + 1*nrecur_, kSetZero);

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

//    void Reset(std::vector<int> &reset_flag) {
//        KALDI_ASSERT(reset_flag.size() == 1);
//        if (reset_flag[0] == 1) {
//            prev_nnet_state_.SetZero();
//        }
//    }

    void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
        int DEBUG = 0;
        int32 T = in.NumRows();
        // resize & clear propagate buffers
        propagate_buf_.Resize(T+2, 7 * ncell_ + nrecur_, kSetZero);  // 0:forward pass history, [1, T]:current sequence, T+1:dummy

	//// The reason I commented out the following line:
	//// Now I make this component completely compatible with "nnet-train-perutt"
	//// so each batch is actually a sentence, history should not be bridged between sentences
        //propagate_buf_.Row(0).CopyFromVec(prev_nnet_state_);

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

        // (x & bias) -> g, i, f, o, not recurrent, do it all in once
        YGIFO.RowRange(1,T).AddMatMat(1.0, in, kNoTrans, w_gifo_x_, kTrans, 0.0);
        YGIFO.RowRange(1,T).AddVecToRows(1.0, bias_);

        for (int t = 1; t <= T; t++) {
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
    
            // recursion r(t-1) -> g, i, f, o
            y_gifo.AddMatVec(1.0, w_gifo_r_, kNoTrans, YR.Row(t-1), 1.0);
            // peephole c(t-1) -> i(t)
            y_i.AddVecVec(1.0, peephole_i_c_, YC.Row(t-1), 1.0);
            // peephole c(t-1) -> f(t)
            y_f.AddVecVec(1.0, peephole_f_c_, YC.Row(t-1), 1.0);

            // i, f sigmoid squashing
            YI_t.Sigmoid(YI_t);
            YF_t.Sigmoid(YF_t);
    
            // g tanh squashing
            YG_t.Tanh(YG_t);
    
            // c memory cell
            y_c.AddVecVec(1.0, y_i, y_g, 0.0);
            // CEC connection via forget gate: c(t-1) -> c(t)
            y_c.AddVecVec(1.0, y_f, YC.Row(t-1), 1.0);

            YC_t.ApplyFloor(-50);   // optional clipping of cell activation
            YC_t.ApplyCeiling(50);  // google paper Interspeech2014: LSTM for LVCSR
    
            // h tanh squashing
            YH_t.Tanh(YC_t);
    
            // o output gate
            y_o.AddVecVec(1.0, peephole_o_c_, y_c, 1.0);  // notice: output gate peephole is not recurrent
            YO_t.Sigmoid(YO_t);
    
            // m
            y_m.AddVecVec(1.0, y_o, y_h, 0.0);
            
            // r
            y_r.AddMatVec(1.0, w_r_m_, kNoTrans, y_m, 0.0);

            if (DEBUG) {
                std::cerr << "forward-pass frame " << t << "\n";
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
        out->CopyFromMat(YR.RowRange(1,T));

        //// now the last frame state becomes previous network state for next batch
        //prev_nnet_state_.CopyFromVec(propagate_buf_.Row(T));
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
        backpropagate_buf_.Resize(T+2, 7 * ncell_ + nrecur_, kSetZero);  // 0:dummy, [1,T] frames, T+1 backward pass history

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

        // projection layer to LSTM output is not recurrent, so backprop it all in once
        DR.RowRange(1,T).CopyFromMat(out_diff);

        for (int t = T; t >= 1; t--) {
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
    
            // r
            //   Version 1 (precise gradients): 
            //   backprop error from g(t+1), i(t+1), f(t+1), o(t+1) to r(t)
            d_r.AddMatVec(1.0, w_gifo_r_, kTrans, DGIFO.Row(t+1), 1.0);

            /*
            //   Version 2 (Alex Graves' PhD dissertation): 
            //   only backprop g(t+1) to r(t) 
            CuSubMatrix<BaseFloat> w_g_r_(w_gifo_r_.RowRange(0, ncell_));
            d_r.AddMatVec(1.0, w_g_r_, kTrans, DG.Row(t+1), 1.0);
            */

            /*
            //   Version 3 (Felix Gers' PhD dissertation): 
            //   truncate gradients of g(t+1), i(t+1), f(t+1), o(t+1) once they leak out memory block
            //   CEC(with forget connection) is the only "error-bridge" through time
            ;
            */
    
            // m
            d_m.AddMatVec(1.0, w_r_m_, kTrans, d_r, 0.0);
    
            // h
            d_h.AddVecVec(1.0, y_o, d_m, 0.0);
            DH_t.DiffTanh(YH_t, DH_t);
    
            // o
            d_o.AddVecVec(1.0, y_h, d_m, 0.0);
            DO_t.DiffSigmoid(YO_t, DO_t);
    
            // c
            //   1. diff from h(t)
            //   2. diff from o(t) (via peephole)
            //   3. diff from c(t+1) (via forget-gate between CEC)
            //   4. diff from f(t+1) (via peephole)
            //   5. diff from i(t+1) (via peephole)
            d_c.AddVec(1.0, d_h, 0.0);  
            d_c.AddVecVec(1.0, peephole_o_c_,  d_o, 1.0);
            d_c.AddVecVec(1.0, YF.Row(t+1),    DC.Row(t+1), 1.0);
            d_c.AddVecVec(1.0, peephole_f_c_ , DF.Row(t+1), 1.0);
            d_c.AddVecVec(1.0, peephole_i_c_,  DI.Row(t+1), 1.0);
    
            // f
            d_f.AddVecVec(1.0, YC.Row(t-1), d_c, 0.0);
            DF_t.DiffSigmoid(YF_t, DF_t);
    
            // i
            d_i.AddVecVec(1.0, y_g, d_c, 0.0);
            DI_t.DiffSigmoid(YI_t, DI_t);
    
            // g
            d_g.AddVecVec(1.0, y_i, d_c, 0.0);
            DG_t.DiffTanh(YG_t, DG_t);
    
            // debug info
            if (DEBUG) {
                std::cerr << "backward-pass frame " << t << "\n";
                //std::cerr << "derivative w.r.t input y " << out_diff.RowRange(t,1);
                std::cerr << "derivative wrt input r " << d_r;
                std::cerr << "derivative wrt input m " << d_m;
                std::cerr << "derivative wrt input h " << d_h;
                std::cerr << "derivative wrt input o " << d_o;
                std::cerr << "derivative wrt input c " << d_c;
                std::cerr << "derivative wrt input f " << d_f;
                std::cerr << "derivative wrt input i " << d_i;
                std::cerr << "derivative wrt input g " << d_g;
            }
        }

        // backprop derivatives to input x, do it all in once
        in_diff->AddMatMat(1.0, DGIFO.RowRange(1,T), kNoTrans, w_gifo_x_, kNoTrans, 0.0);
    
        // calculate delta
        const BaseFloat mmt = opts_.momentum;
    
        w_gifo_x_corr_.AddMatMat(1.0, DGIFO.RowRange(1,T), kTrans, in              , kNoTrans, mmt);
        w_gifo_r_corr_.AddMatMat(1.0, DGIFO.RowRange(1,T), kTrans, YR.RowRange(0,T), kNoTrans, mmt);  // recurrent r -> g

        bias_corr_.AddRowSumMat(1.0, DGIFO.RowRange(1,T), mmt);
    
        peephole_i_c_corr_.AddDiagMatMat(1.0, DI.RowRange(1,T), kTrans, YC.RowRange(0,T), kNoTrans, mmt);  // recurrent c -> i
        peephole_f_c_corr_.AddDiagMatMat(1.0, DF.RowRange(1,T), kTrans, YC.RowRange(0,T), kNoTrans, mmt);  // recurrent c -> f
        peephole_o_c_corr_.AddDiagMatMat(1.0, DO.RowRange(1,T), kTrans, YC.RowRange(1,T), kNoTrans, mmt);

        w_r_m_corr_.AddMatMat(1.0, DR.RowRange(1,T), kTrans, YM.RowRange(1,T), kNoTrans, mmt);
    
        if (DEBUG) {
            std::cerr << "gradients(with optional momentum): \n";
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
        const BaseFloat lr  = opts_.learn_rate;

        w_gifo_x_.AddMat(-lr, w_gifo_x_corr_);
        w_gifo_r_.AddMat(-lr, w_gifo_r_corr_);
        bias_.AddVec(-lr, bias_corr_, 1.0);
    
        peephole_i_c_.AddVec(-lr, peephole_i_c_corr_, 1.0);
        peephole_f_c_.AddVec(-lr, peephole_f_c_corr_, 1.0);
        peephole_o_c_.AddVec(-lr, peephole_o_c_corr_, 1.0);
    
        w_r_m_.AddMat(-lr, w_r_m_corr_);

//        /* 
//          Here we deal with the famous "vanishing & exploding difficulties" in RNN learning.
//
//          *For gradients vanishing*
//            LSTM architecture introduces linear CEC as the "error bridge" across long time distance
//            solving vanishing problem.
//
//          *For gradients exploding*
//            LSTM is still vulnerable to gradients explosing in BPTT(with large weight & deep time expension).
//            To prevent this, we tried L2 regularization, which didn't work well
//
//          Our approach is a *modified* version of Max Norm Regularization:
//          For each nonlinear neuron, 
//            1. fan-in weights & bias model a seperation hyper-plane: W x + b = 0
//            2. squashing function models a differentiable nonlinear slope around this hyper-plane.
//
//          Conventional max norm regularization scale W to keep its L2 norm bounded,
//          As a modification, we scale down large (W & b) *simultaneously*, this:
//            1. keeps all fan-in weights small, prevents gradients from exploding during backward-pass.
//            2. keeps the location of the hyper-plane unchanged, so we don't wipe out already learned knowledge.
//            3. shrinks the "normal" of the hyper-plane, smooths the nonlinear slope, improves generalization.
//            4. makes the network *well-conditioned* (weights are constrained in a reasonible range).
//
//          We've observed faster convergence and performance gain by doing this.
//        */
//
//        int DEBUG = 0;
//        BaseFloat max_norm = 1.0;   // weights with large L2 norm may cause exploding in deep BPTT expensions
//                                    // TODO: move this config to opts_
//        CuMatrix<BaseFloat> L2_gifo_x(w_gifo_x_);
//        CuMatrix<BaseFloat> L2_gifo_r(w_gifo_r_);
//        L2_gifo_x.MulElements(w_gifo_x_);
//        L2_gifo_r.MulElements(w_gifo_r_);
//
//        CuVector<BaseFloat> L2_norm_gifo(L2_gifo_x.NumRows());
//        L2_norm_gifo.AddColSumMat(1.0, L2_gifo_x, 0.0);
//        L2_norm_gifo.AddColSumMat(1.0, L2_gifo_r, 1.0);
//        L2_norm_gifo.Range(1*ncell_, ncell_).AddVecVec(1.0, peephole_i_c_, peephole_i_c_, 1.0);
//        L2_norm_gifo.Range(2*ncell_, ncell_).AddVecVec(1.0, peephole_f_c_, peephole_f_c_, 1.0);
//        L2_norm_gifo.Range(3*ncell_, ncell_).AddVecVec(1.0, peephole_o_c_, peephole_o_c_, 1.0);
//        L2_norm_gifo.ApplyPow(0.5);
//
//        CuVector<BaseFloat> shrink(L2_norm_gifo);
//        shrink.Scale(1.0/max_norm);
//        shrink.ApplyFloor(1.0);
//        shrink.InvertElements();
//
//        w_gifo_x_.MulRowsVec(shrink);
//        w_gifo_r_.MulRowsVec(shrink);
//        bias_.MulElements(shrink);
//
//        peephole_i_c_.MulElements(shrink.Range(1*ncell_, ncell_));
//        peephole_f_c_.MulElements(shrink.Range(2*ncell_, ncell_));
//        peephole_o_c_.MulElements(shrink.Range(3*ncell_, ncell_));
//
//        if (DEBUG) {
//            if (shrink.Min() < 0.95) {   // we dont want too many trivial logs here
//                std::cerr << "gifo shrinking coefs: " << shrink;
//            }
//        }
//        
    }

private:
    // dims
    int32 ncell_;
    int32 nrecur_;  // recurrent projection layer dim

    //
    //CuVector<BaseFloat> prev_nnet_state_;

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

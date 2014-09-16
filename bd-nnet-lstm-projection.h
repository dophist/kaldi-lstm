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
        input_dim_(input_dim),
        output_dim_(output_dim),
        inverse_(false),
        ncell_(0),
        nrecur_(0)
    { }

    ~LstmProjection()
    { }

    Component* Copy() const { return new LstmProjection(*this); }
    ComponentType GetType() const { return kLstmProjection; }

    void InitData(std::istream &is) {
        // define options
        float param_stddev = 0.1;
        // parse config
        std::string token;
        while (!is.eof()) {
            ReadToken(is, false, &token); 
            if(token == "<Inverse>") 
                ReadBasicType(is, false, &inverse_);
            else if (token == "<NumCell>") 
                ReadBasicType(is, false, &ncell_);
            else if (token == "<NumRecur>") 
                ReadBasicType(is, false, &nrecur_);
            else if (token == "<ParamStddev>") 
                ReadBasicType(is, false, &param_stddev);
            else if (token == "") 
                ReadBasicType(is, false, &param_stddev);
            else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                           << " (Inverse|NumCell|NumRecur|ParamStddev)";
            is >> std::ws; // eat-up whitespace
        }

        w_gifo_x_.Resize(4*ncell_, input_dim_); 
        w_gifo_x_.SetRandn();  w_gifo_x_.Scale(param_stddev);

        w_gifo_r_.Resize(4*ncell_, nrecur_);    
        w_gifo_r_.SetRandn();  w_gifo_r_.Scale(param_stddev);

        bias_.Resize(4*ncell_);     
        bias_.SetRandn();  bias_.Scale(param_stddev);

        peephole_i_c_.Resize(ncell_);
        peephole_i_c_.SetRandn();  peephole_i_c_.Scale(param_stddev);

        peephole_f_c_.Resize(ncell_);
        peephole_f_c_.SetRandn();  peephole_f_c_.Scale(param_stddev);

        peephole_o_c_.Resize(ncell_);
        peephole_o_c_.SetRandn();  peephole_o_c_.Scale(param_stddev);

        w_r_m_.Resize(nrecur_, ncell_); 
        w_r_m_.SetRandn();  w_r_m_.Scale(param_stddev);

        // delta buffers need to be initialized during component init (due to momentum)
        w_gifo_x_corr_.Resize(4*ncell_, input_dim_, kSetZero); 
        w_gifo_r_corr_.Resize(4*ncell_, nrecur_, kSetZero);    
        bias_corr_.Resize(4*ncell_, kSetZero);     

        peephole_i_c_corr_.Resize(ncell_, kSetZero);
        peephole_f_c_corr_.Resize(ncell_, kSetZero);
        peephole_o_c_corr_.Resize(ncell_, kSetZero);

        w_r_m_corr_.Resize(nrecur_, ncell_, kSetZero); 
    }

    void ReadData(std::istream &is, bool binary) {
        ReadBasicType(is, binary, &inverse_);
        ReadBasicType(is, binary, &ncell_);
        ReadBasicType(is, binary, &nrecur_);

        // TODO
        KALDI_ASSERT(nrecur_ == output_dim_);

        w_gifo_x_.Read(is, binary);
        w_gifo_r_.Read(is, binary);
        bias_.Read(is, binary);

        peephole_i_c_.Read(is, binary);
        peephole_f_c_.Read(is, binary);
        peephole_o_c_.Read(is, binary);

        w_r_m_.Read(is, binary);

        // delta buffers need to be initialized during model load-in (due to momentum)
        w_gifo_x_corr_.Resize(4*ncell_, input_dim_, kSetZero); 
        w_gifo_r_corr_.Resize(4*ncell_, nrecur_, kSetZero);    
        bias_corr_.Resize(4*ncell_, kSetZero);     

        peephole_i_c_corr_.Resize(ncell_, kSetZero);
        peephole_f_c_corr_.Resize(ncell_, kSetZero);
        peephole_o_c_corr_.Resize(ncell_, kSetZero);

        w_r_m_corr_.Resize(nrecur_, ncell_, kSetZero); 
    }

    void WriteData(std::ostream &os, bool binary) const {
        WriteBasicType(os, binary, inverse_);
        WriteBasicType(os, binary, ncell_);
        WriteBasicType(os, binary, nrecur_);

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
            "\n  w_gifo_x_  " + MomentStatistics(w_gifo_x_) + 
            "\n  w_gifo_r_  " + MomentStatistics(w_gifo_r_) +
            "\n  bias_  " + MomentStatistics(bias_) +

            "\n  peephole_i_c_  " + MomentStatistics(peephole_i_c_) +
            "\n  peephole_f_c_  " + MomentStatistics(peephole_f_c_) +
            "\n  peephole_o_c_  " + MomentStatistics(peephole_o_c_) +

            "\n  w_r_m_  " + MomentStatistics(w_r_m_);
    }
  
    std::string InfoGradient() const {
        return std::string("    ") + 
            "\n  w_gifo_x_corr_  " + MomentStatistics(w_gifo_x_corr_) + 
            "\n  w_gifo_r_corr_  " + MomentStatistics(w_gifo_r_corr_) +
            "\n  bias_corr_  " + MomentStatistics(bias_corr_) +

            "\n  peephole_i_c_corr_  " + MomentStatistics(peephole_i_c_corr_) +
            "\n  peephole_f_c_corr_  " + MomentStatistics(peephole_f_c_corr_) +
            "\n  peephole_o_c_corr_  " + MomentStatistics(peephole_o_c_corr_) +

            "\n  w_r_m_corr_  " + MomentStatistics(w_r_m_corr_);
    }

    void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
        int32 T = in.NumRows();
        // resize & clear propagate buffers
        propagate_buf_.Resize(T, 7 * ncell_ + nrecur_, kSetZero);

        CuSubMatrix<BaseFloat> Y_g(propagate_buf_.ColRange(0*ncell_, ncell_));
        CuSubMatrix<BaseFloat> Y_i(propagate_buf_.ColRange(1*ncell_, ncell_));
        CuSubMatrix<BaseFloat> Y_f(propagate_buf_.ColRange(2*ncell_, ncell_));
        CuSubMatrix<BaseFloat> Y_o(propagate_buf_.ColRange(3*ncell_, ncell_));
        CuSubMatrix<BaseFloat> Y_c(propagate_buf_.ColRange(4*ncell_, ncell_));
        CuSubMatrix<BaseFloat> Y_h(propagate_buf_.ColRange(5*ncell_, ncell_));
        CuSubMatrix<BaseFloat> Y_m(propagate_buf_.ColRange(6*ncell_, ncell_));
        CuSubMatrix<BaseFloat> Y_r(propagate_buf_.ColRange(7*ncell_, nrecur_));
        CuSubMatrix<BaseFloat> Y_gifo(propagate_buf_.ColRange(0, 4*ncell_));

        // propagate x and add bias all in once
        Y_gifo.AddMatMat(1.0, in, kNoTrans, w_gifo_x_, kTrans, 0.0);
        Y_gifo.AddVecToRows(1.0, bias_);

        CuMatrix<BaseFloat> add_buf(1, ncell_, kUndefined);
        for (int t = 0; t < T; t++) {
            CuSubMatrix<BaseFloat> y_g(Y_g.RowRange(t,1));
            CuSubMatrix<BaseFloat> y_i(Y_i.RowRange(t,1));
            CuSubMatrix<BaseFloat> y_f(Y_f.RowRange(t,1));
            CuSubMatrix<BaseFloat> y_o(Y_o.RowRange(t,1));
            CuSubMatrix<BaseFloat> y_c(Y_c.RowRange(t,1));
            CuSubMatrix<BaseFloat> y_h(Y_h.RowRange(t,1));
            CuSubMatrix<BaseFloat> y_m(Y_m.RowRange(t,1));
            CuSubMatrix<BaseFloat> y_r(Y_r.RowRange(t,1));

            CuSubMatrix<BaseFloat> y_gifo(Y_gifo.RowRange(t,1));
    
            if (t > 0) {
                // recursion r(t-1) to g, i, f, o
                y_gifo.AddMatMat(1.0, Y_r.RowRange(t-1,1), kNoTrans, w_gifo_r_, kTrans, 1.0);
                // peephole c(t-1) to i(t)
                add_buf.CopyFromMat(Y_c.RowRange(t-1,1)); 
                add_buf.MulColsVec(peephole_i_c_);
                y_i.AddMat(1.0, add_buf);
                // peephole c(t-1) to f(t)
                add_buf.CopyFromMat(Y_c.RowRange(t-1,1));
                add_buf.MulColsVec(peephole_f_c_);
                y_f.AddMat(1.0, add_buf);
            }

            // i, f sigmoid squashing
            y_i.Sigmoid(y_i);
            y_f.Sigmoid(y_f);
    
            // g tanh squashing
            y_g.Tanh(y_g);
    
            // c memory cell
            if (t > 0) {  // CEC connection via forget gate
                y_c.CopyFromMat(Y_c.RowRange(t-1,1));
                y_c.MulElements(y_f);
            }

            add_buf.CopyFromMat(y_g); 
            add_buf.MulElements(y_i);
            y_c.AddMat(1.0, add_buf);

            // optional clipping of cell activation (google paper Interspeech2014)
            y_c.ApplyFloor(-50);
            y_c.ApplyCeiling(50);
    
            // h tanh squashing
            y_h.CopyFromMat(y_c);
            y_h.Tanh(y_h);
    
            // o output gate
            add_buf.CopyFromMat(y_c);
            add_buf.MulColsVec(peephole_o_c_);
            y_o.AddMat(1.0, add_buf);
            y_o.Sigmoid(y_o);
    
            // m
            y_m.CopyFromMat(y_h);
            y_m.MulElements(y_o);
            
            // r
            y_r.AddMatMat(1.0, y_m, kNoTrans, w_r_m_, kTrans, 0.0);

            // TODO: add non-recursive node in projection layer (projection layer is full recursive for now)
            out->RowRange(t,1).CopyFromMat(y_r);

            //// debug info
            //std::cerr << "forward frame " << t << "\n";
            //std::cerr << "y_g " << y_g;
            //std::cerr << "y_i " << y_i;
            //std::cerr << "y_f " << y_f;
            //std::cerr << "y_o " << y_o;
            //std::cerr << "y_gifo " << y_gifo;
            //std::cerr << "y_c " << y_c;
            //std::cerr << "y_h " << y_h;
            //std::cerr << "y_m " << y_m;
            //std::cerr << "y_r " << y_r;
            //std::cerr << "y " << y;
        }
    }

    void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                            const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
        int32 T = in.NumRows();
    
        // propagated buffers already computed
        CuSubMatrix<BaseFloat> Y_g(propagate_buf_.ColRange(0*ncell_, ncell_));
        CuSubMatrix<BaseFloat> Y_i(propagate_buf_.ColRange(1*ncell_, ncell_));
        CuSubMatrix<BaseFloat> Y_f(propagate_buf_.ColRange(2*ncell_, ncell_));
        CuSubMatrix<BaseFloat> Y_o(propagate_buf_.ColRange(3*ncell_, ncell_));
        CuSubMatrix<BaseFloat> Y_c(propagate_buf_.ColRange(4*ncell_, ncell_));
        CuSubMatrix<BaseFloat> Y_h(propagate_buf_.ColRange(5*ncell_, ncell_));
        CuSubMatrix<BaseFloat> Y_m(propagate_buf_.ColRange(6*ncell_, ncell_));
        CuSubMatrix<BaseFloat> Y_r(propagate_buf_.ColRange(7*ncell_, nrecur_));
    
        // backpropagate buffers to compute
        backpropagate_buf_.Resize(T, 7 * ncell_ + nrecur_, kSetZero);

        CuSubMatrix<BaseFloat> D_g(backpropagate_buf_.ColRange(0*ncell_, ncell_));
        CuSubMatrix<BaseFloat> D_i(backpropagate_buf_.ColRange(1*ncell_, ncell_));
        CuSubMatrix<BaseFloat> D_f(backpropagate_buf_.ColRange(2*ncell_, ncell_));
        CuSubMatrix<BaseFloat> D_o(backpropagate_buf_.ColRange(3*ncell_, ncell_));
        CuSubMatrix<BaseFloat> D_c(backpropagate_buf_.ColRange(4*ncell_, ncell_));
        CuSubMatrix<BaseFloat> D_h(backpropagate_buf_.ColRange(5*ncell_, ncell_));
        CuSubMatrix<BaseFloat> D_m(backpropagate_buf_.ColRange(6*ncell_, ncell_));
        CuSubMatrix<BaseFloat> D_r(backpropagate_buf_.ColRange(7*ncell_, nrecur_));

        CuSubMatrix<BaseFloat> D_gifo(backpropagate_buf_.ColRange(0, 4*ncell_));
    
        CuMatrix<BaseFloat> add_buf(1, ncell_, kUndefined);
        for (int t = T-1; t >= 0; t--) {
            CuSubMatrix<BaseFloat> y_g(Y_g.RowRange(t,1));
            CuSubMatrix<BaseFloat> y_i(Y_i.RowRange(t,1));
            CuSubMatrix<BaseFloat> y_f(Y_f.RowRange(t,1));
            CuSubMatrix<BaseFloat> y_o(Y_o.RowRange(t,1));
            CuSubMatrix<BaseFloat> y_c(Y_c.RowRange(t,1));
            CuSubMatrix<BaseFloat> y_h(Y_h.RowRange(t,1));
            CuSubMatrix<BaseFloat> y_m(Y_m.RowRange(t,1));
            CuSubMatrix<BaseFloat> y_r(Y_r.RowRange(t,1));
    
            CuSubMatrix<BaseFloat> d_g(D_g.RowRange(t,1));
            CuSubMatrix<BaseFloat> d_i(D_i.RowRange(t,1));
            CuSubMatrix<BaseFloat> d_f(D_f.RowRange(t,1));
            CuSubMatrix<BaseFloat> d_o(D_o.RowRange(t,1));
            CuSubMatrix<BaseFloat> d_c(D_c.RowRange(t,1));
            CuSubMatrix<BaseFloat> d_h(D_h.RowRange(t,1));
            CuSubMatrix<BaseFloat> d_m(D_m.RowRange(t,1));
            CuSubMatrix<BaseFloat> d_r(D_r.RowRange(t,1));
    
            CuSubMatrix<BaseFloat> d_gifo(D_gifo.RowRange(t,1));
    
            // r
            d_r.CopyFromMat(out_diff.RowRange(t,1));
    
            if (t < T-1) {
                // in Alex Grave's PhD dissertation,
                // he didn't backpropagate error from i(t+1), f(t+1), o(t+1) to r(t),
                // we compute precise gradient here anyway
                d_r.AddMatMat(1.0, D_gifo.RowRange(t+1,1), kNoTrans, w_gifo_r_, kNoTrans, 1.0);
            }
    
            // m
            d_m.AddMatMat(1.0, d_r, kNoTrans, w_r_m_, kNoTrans, 0.0);
    
            // h
            d_h.CopyFromMat(d_m); d_h.MulElements(y_o);
            d_h.DiffTanh(y_h, d_h);
    
            // o
            d_o.CopyFromMat(d_m); d_o.MulElements(y_h);
            d_o.DiffSigmoid(y_o, d_o);
    
            // c
            //     1. diff from h(t)
            d_c.CopyFromMat(d_h);  
            //     2. diff from output-gate(t) (via peephole)
            add_buf.CopyFromMat(d_o); add_buf.MulColsVec(peephole_o_c_);  
            d_c.AddMat(1.0, add_buf);  
    
            if (t < T-1) {
                // 3. diff from c(t+1) (via forget-gate between CEC)
                add_buf.CopyFromMat(D_c.RowRange(t+1,1)); add_buf.MulElements(Y_f.RowRange(t+1,1));  
                d_c.AddMat(1.0, add_buf);  
                // 4. diff from forget-gate(t+1) (via peephole)
                add_buf.CopyFromMat(D_f.RowRange(t+1,1)); add_buf.MulColsVec(peephole_f_c_);  
                d_c.AddMat(1.0, add_buf);
                // 5. diff from input-gate(t+1) (via peephole)
                add_buf.CopyFromMat(D_i.RowRange(t+1,1)); add_buf.MulColsVec(peephole_i_c_);  
                d_c.AddMat(1.0, add_buf);
            }
    
            // f
            if (t > 0) {
                d_f.CopyFromMat(d_c);
                d_f.MulElements(Y_c.RowRange(t-1,1));
                d_f.DiffSigmoid(y_f, d_f);
            }
    
            // i
            d_i.CopyFromMat(d_c);
            d_i.MulElements(y_g);
            d_i.DiffSigmoid(y_i, d_i);
    
            // g
            d_g.CopyFromMat(d_c);
            d_g.MulElements(y_i);
            d_g.DiffTanh(y_g, d_g);
    
            // x
            in_diff->RowRange(t,1).AddMatMat(1.0, d_gifo, kNoTrans, w_gifo_x_, kNoTrans, 0.0);
    
            //// debug info
            //std::cerr << "backward frame " << t << "\n";
            //std::cerr << "d_y " << out_diff.RowRange(t,1);
            //std::cerr << "d_r " << d_r;
            //std::cerr << "d_m " << d_m;
            //std::cerr << "d_h " << d_h;
            //std::cerr << "d_o " << d_o;
            //std::cerr << "d_c " << d_c;
            //std::cerr << "d_f " << d_f;
            //std::cerr << "d_i " << d_i;
            //std::cerr << "d_g " << d_g;
            //std::cerr << "d_gifo " << d_gifo;
            //std::cerr << "d_x " << in_diff->RowRange(t,1);
        }
    
        // calculate delta
        const BaseFloat lr = opts_.learn_rate;
        const BaseFloat mmt = opts_.momentum;
    
        w_gifo_x_corr_.AddMatMat(-lr, D_gifo, kTrans, in, kNoTrans, mmt);
        w_gifo_r_corr_.AddMatMat(-lr, D_gifo.RowRange(1,T-1), kTrans, Y_r.RowRange(0,T-1), kNoTrans, mmt);
        bias_corr_.AddRowSumMat(-lr, D_gifo, mmt);
    
        w_r_m_corr_.AddMatMat(-lr, D_r, kTrans, Y_m, kNoTrans, mmt);
    
        CuMatrix<BaseFloat> buf(T, ncell_, kUndefined);
    
        buf.SetZero();
        buf.RowRange(0,T-1).CopyFromMat(Y_c.RowRange(0,T-1));
        buf.RowRange(0,T-1).MulElements(D_i.RowRange(1, T-1));
        peephole_i_c_corr_.AddRowSumMat(-lr, buf.RowRange(0,T-1), mmt);
    
        buf.SetZero();
        buf.RowRange(0,T-1).CopyFromMat(Y_c.RowRange(0,T-1));
        buf.RowRange(0,T-1).MulElements(D_f.RowRange(1, T-1));
        peephole_f_c_corr_.AddRowSumMat(-lr, buf.RowRange(0,T-1), mmt);
    
        buf.SetZero();
        buf.CopyFromMat(Y_c);
        buf.MulElements(D_o);
        peephole_o_c_corr_.AddRowSumMat(-lr, buf, mmt);
    
        //// debug info
        //std::cerr << "delta: \n";
        //std::cerr << "w_gifo_x_corr_ " << w_gifo_x_corr_;
        //std::cerr << "w_gifo_r_corr_ " << w_gifo_r_corr_;
        //std::cerr << "bias_corr_ " << bias_corr_;
        //std::cerr << "w_r_m_corr_ " << w_r_m_corr_;
        //std::cerr << "peephole_i_c_corr_ " << peephole_i_c_corr_;
        //std::cerr << "peephole_f_c_corr_ " << peephole_f_c_corr_;
        //std::cerr << "peephole_o_c_corr_ " << peephole_o_c_corr_;
    
    }

void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
    w_gifo_x_.AddMat(1.0, w_gifo_x_corr_);
    w_gifo_r_.AddMat(1.0, w_gifo_r_corr_);
    bias_.AddVec(1.0, bias_corr_);

    w_r_m_.AddMat(1.0, w_r_m_corr_);

    peephole_i_c_.AddVec(1.0, peephole_i_c_corr_);
    peephole_f_c_.AddVec(1.0, peephole_f_c_corr_);
    peephole_o_c_.AddVec(1.0, peephole_o_c_corr_);
}

private:
    /* network topology parameters */
    int32 input_dim_;
    int32 output_dim_;
    bool inverse_;
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

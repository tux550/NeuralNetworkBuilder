#ifndef H_FCLAYER
#define H_FCLAYER

#include <random>
#include <iostream>
#include "../../algebra/alg.h"
#include"./layer.h"

namespace ai{
    class FCLayer: public BaseLayer
    {
    private:
        alg::t_mat weights_mat;
        alg::t_mat bias;
    public:
        FCLayer(alg::t_dim inp_size, alg::t_dim out_size);
        alg::t_mat forward_propagation_implementation(alg::t_mat &im);
        alg::t_mat backward_propagation_implementation(alg::t_mat &out_error, alg::t_type alpha);
        void set_weights(alg::t_mat &w);
        alg::t_mat get_weights();
        // Write
        void write(std::ostream& os);
        void read(std::istream& is);
    };
}

#endif
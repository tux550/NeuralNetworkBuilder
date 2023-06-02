#ifndef H_FCLAYER
#define H_FCLAYER

#include <random>
#include "../algebra/alg.h"
#include"./layer.h"

namespace ai{
    class FCLayer: public BaseLayer
    {
    private:
        alg::Matrix weights_mat;
        alg::Matrix bias;
    public:
        FCLayer(alg::t_dim inp_size, alg::t_dim out_size);
        alg::Matrix forward_propagation_implementation(alg::Matrix &im);
        alg::Matrix backward_propagation(alg::Matrix &out_error, alg::t_type alpha);
        void set_weights(alg::Matrix &w);
        alg::Matrix get_weights();
    };
}

#endif
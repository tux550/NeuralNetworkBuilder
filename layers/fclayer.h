#ifndef H_FCLAYER
#define H_FCLAYER

#include <random>
#include "../algebra/alg.h"
#include"./layer.h"

namespace ai{
    class FCLayer: public BaseLayer
    {
    private:
        alg::MultidimMatrix weights_mat;
        alg::MultidimMatrix bias;
    public:
        FCLayer(alg::t_dim inp_size, alg::t_dim out_size);
        alg::MultidimMatrix forward_propagation_implementation(alg::MultidimMatrix &im);
        alg::MultidimMatrix backward_propagation(alg::MultidimMatrix &out_error, alg::t_type alpha);
        void set_weights(alg::MultidimMatrix &w);
        alg::MultidimMatrix get_weights();
    };
}

#endif
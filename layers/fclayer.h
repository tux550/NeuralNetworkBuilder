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
    public:
        FCLayer(alg::t_dim inp_size, alg::t_dim out_size);
        void forward_propagation();
        void backward_propagation();
        void set_weights(alg::Matrix w);
    };
}

#endif
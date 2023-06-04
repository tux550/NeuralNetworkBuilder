#ifndef H_ACTLAYER
#define H_ACTLAYER

#include <functional>
#include "../algebra/alg.h"
#include"./layer.h"

namespace ai{
    class ActLayer: public BaseLayer
    {
        private:
            alg::t_t2t act_func;
            alg::t_t2t drv_func;
        public:
            ActLayer(alg::t_dim n_size, alg::t_t2t _act_func, alg::t_t2t _drv_func);
            alg::MultidimMatrix forward_propagation_implementation(alg::MultidimMatrix &im);
            alg::MultidimMatrix backward_propagation(alg::MultidimMatrix &out_error, alg::t_type alpha);
    };
};

#endif
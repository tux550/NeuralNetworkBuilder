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
        public:
            ActLayer(alg::t_dim n_size, alg::t_t2t _act_func);
            void forward_propagation();
            void backward_propagation();
    };
};

#endif
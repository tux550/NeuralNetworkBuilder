#ifndef H_ACTLAYER
#define H_ACTLAYER

#include <functional>
#include "../../algebra/alg.h"
#include"./layer.h"

namespace ai{
    class ActLayer: public BaseLayer
    {
        private:
            alg::t_t2t act_func;
            alg::t_t2t drv_func;
        public:
            ActLayer(alg::t_dim n_size, alg::t_t2t _act_func, alg::t_t2t _drv_func);
            alg::Matrix forward_propagation_implementation(alg::Matrix &im);
            alg::Matrix backward_propagation_implementation(alg::Matrix &data, alg::Matrix &out_error, alg::t_type alpha);
            void write(std::ostream& os) const;
    };
};

#endif
#ifndef H_ACTLAYER
#define H_ACTLAYER

#include <functional>
#include "../../algebra/alg.h"
#include"./layer.h"

namespace ai{
    class ActLayer: public BaseLayer
    {
        private:
            alg::t_m2m act_func;
            alg::t_m2m drv_func;
        public:
            ActLayer(alg::t_dim n_size, alg::t_m2m _act_func, alg::t_m2m _drv_func);
            alg::t_mat forward_propagation_implementation(alg::t_mat &im);
            alg::t_mat backward_propagation_implementation(alg::t_mat &out_error, alg::t_type alpha);
            void write(std::ostream& os) const;
            void read(std::istream& is);
    };
};

#endif
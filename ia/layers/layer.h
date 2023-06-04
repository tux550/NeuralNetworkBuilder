#ifndef H_LAYER
#define H_LAYER

#include "../../algebra/alg.h"

namespace ai
{
    class BaseLayer
    {
    protected:
        alg::Matrix input_data;
        alg::t_dim input_size;
        alg::t_dim output_size;
    public:
        BaseLayer(alg::t_dim inp_size, alg::t_dim out_size);
        alg::Matrix forward_propagation(alg::Matrix &im);
        virtual alg::Matrix forward_propagation_implementation(alg::Matrix &im)  = 0;
        virtual alg::Matrix backward_propagation(alg::Matrix &out_error, alg::t_type alpha) = 0;
        
    };
}



#endif

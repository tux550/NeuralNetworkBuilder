#ifndef H_LAYER
#define H_LAYER

#include "../algebra/alg.h"

namespace ai
{
    class BaseLayer
    {
    protected:
        alg::MultidimMatrix input_data;
        alg::t_dim input_size;
        alg::t_dim output_size;
    public:
        BaseLayer(alg::t_dim inp_size, alg::t_dim out_size);
        alg::MultidimMatrix forward_propagation(alg::MultidimMatrix &im);
        virtual alg::MultidimMatrix forward_propagation_implementation(alg::MultidimMatrix &im)  = 0;
        virtual alg::MultidimMatrix backward_propagation(alg::MultidimMatrix &out_error, alg::t_type alpha) = 0;
        
    };
}



#endif

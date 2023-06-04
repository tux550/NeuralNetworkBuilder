#ifndef H_LAYER
#define H_LAYER

#include "../../algebra/alg.h"

namespace ai
{
    class BaseLayer
    {
    protected:
        alg::vec_mat input_data;
        alg::t_dim   input_size;
        alg::t_dim   output_size;
    public:
        BaseLayer(alg::t_dim inp_size, alg::t_dim out_size);
        
        // Bulk update
        alg::vec_mat forward_propagation(alg::vec_mat &vec_im);
        alg::vec_mat backward_propagation(alg::vec_mat &vec_out_error, alg::t_type alpha);
        // Update base on one result
        virtual alg::Matrix forward_propagation_implementation(alg::Matrix &im) = 0;
        virtual alg::Matrix backward_propagation_implementation(alg::Matrix &data, alg::Matrix &out_error, alg::t_type alpha) = 0;     
    };
}



#endif

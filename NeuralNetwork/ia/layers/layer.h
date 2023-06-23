#ifndef H_LAYER
#define H_LAYER

#include <iostream>
#include <armadillo>
#include "../../algebra/alg.h"

namespace ai
{
    class BaseLayer
    {
    protected:
        alg::t_mat   input_data;
        alg::t_dim   input_size;
        alg::t_dim   output_size;
    public:
        BaseLayer(alg::t_dim inp_size, alg::t_dim out_size);
        
        // Generic propagation
        alg::t_mat forward_propagation(alg::t_mat &im);
        alg::t_mat backward_propagation(alg::t_mat &out_error, alg::t_type alpha);
        // Implementation of propagation
        virtual alg::t_mat forward_propagation_implementation(alg::t_mat &im) = 0;
        virtual alg::t_mat backward_propagation_implementation(alg::t_mat &out_error, alg::t_type alpha) = 0;  
        // Write & Read
        virtual void write(std::ostream& os) = 0;
        virtual void read(std::istream& os) = 0;
    };

    std::ostream& operator<<(std::ostream& os, BaseLayer& layer);
    std::istream& operator>> (std::istream& is, BaseLayer&  dt);
}



#endif

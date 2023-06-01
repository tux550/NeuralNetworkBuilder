#ifndef H_LAYER
#define H_LAYER

#include "../algebra/alg.h"

namespace ai
{
    class BaseLayer
    {
    protected:
        alg::Matrix input_mat;
        alg::Matrix output_mat;
    public:
        BaseLayer(alg::t_dim inp_size, alg::t_dim out_size);
        virtual void forward_propagation()  = 0;
        virtual void backward_propagation() = 0;
        // Getters
        void set_input(alg::Matrix &im);
        // Setters
        alg::Matrix get_output();
    };
}



#endif

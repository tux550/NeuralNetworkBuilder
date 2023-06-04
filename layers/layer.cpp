#include "./fclayer.h"

namespace ai{
    BaseLayer::BaseLayer(alg::t_dim inp_size, alg::t_dim out_size):
        input_data{ {1,inp_size} },
        input_size{inp_size},
        output_size{out_size}
        {}
    
    alg::MultidimMatrix BaseLayer::forward_propagation(alg::MultidimMatrix &im){
        input_data = im;
        return forward_propagation_implementation(im);
    }
}



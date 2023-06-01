#include "./fclayer.h"

namespace ai{

    BaseLayer::BaseLayer(alg::t_dim inp_size, alg::t_dim out_size):
        input_data{1,input_size},
        input_size{inp_size},
        output_size{out_size}
        {}
    
    alg::Matrix BaseLayer::forward_propagation(alg::Matrix &im){
        input_data = im;
        return forward_propagation_implementation(im);
    }
}



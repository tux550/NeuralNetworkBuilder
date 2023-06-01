#include "./fclayer.h"

namespace ai{

    BaseLayer::BaseLayer(alg::t_dim inp_size, alg::t_dim out_size):
        input_mat{1, inp_size},
        output_mat{1,inp_size}
        {}

    // Getters
    void BaseLayer::set_input(alg::Matrix &im) {
        input_mat = im;
    }
    // Setters
    alg::Matrix BaseLayer::get_output() {
        return output_mat;
    }
}



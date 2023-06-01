#include "./actlayer.h"

namespace ai{
    // Constructor
    ActLayer::ActLayer(alg::t_dim n_size, std::function<alg::t_type(alg::t_type)> _act_func):
        BaseLayer{n_size,n_size},
        act_func{_act_func}
        {}
    
    // Forward Propagation
    void ActLayer::forward_propagation() {
        output_mat = input_mat.apply(act_func);
    }

    // Backward Propagation
    void ActLayer::backward_propagation() {
        // TODO
    }
}

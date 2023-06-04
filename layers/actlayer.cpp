#include "./actlayer.h"

namespace ai{
    // Constructor
    ActLayer::ActLayer(alg::t_dim n_size, std::function<alg::t_type(alg::t_type)> _act_func, alg::t_t2t _drv_func):
        BaseLayer{n_size,n_size},
        act_func{_act_func},
        drv_func{_drv_func}
        {}
    
    // Forward Propagation
    alg::MultidimMatrix ActLayer::forward_propagation_implementation(alg::MultidimMatrix &im) {
        return im.apply(act_func);
    }

    // Backward Propagation
    alg::MultidimMatrix ActLayer::backward_propagation(alg::MultidimMatrix &out_error, alg::t_type alpha) {
        return input_data.apply(drv_func) * out_error; // df(input) @ output_error
    }
}

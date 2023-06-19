#include "./actlayer.h"

namespace ai{
    // Constructor
    ActLayer::ActLayer(alg::t_dim n_size, alg::t_m2m _act_func, alg::t_m2m _drv_func):
        BaseLayer{n_size,n_size},
        act_func{_act_func},
        drv_func{_drv_func}
        {}
    
    // Forward Propagation
    alg::t_mat ActLayer::forward_propagation_implementation(alg::t_mat &im) {
        return act_func(im);
    }

    // Backward Propagation
    alg::t_mat ActLayer::backward_propagation_implementation(alg::t_mat &out_error, alg::t_type alpha) {
        // f'(input) @ output_error
        alg::t_mat drv_in = drv_func(input_data); 
        alg::t_mat in_error = drv_in % out_error; // f'(input) @ output_error
        /*
        std::cout << "ACTIVATION"<< std::endl
                 << "out error:" << arma::size(out_error) << std::endl
                 << "drv in:" << arma::size(drv_in) << std::endl
                 << "in error:" << arma::size(in_error) << std::endl;
        */
        return in_error; 
    }

    // Write
    void ActLayer::write(std::ostream& os) const {
        os << std::endl;
    }
    void ActLayer::read(std::istream& is) {
        return;
    }
}

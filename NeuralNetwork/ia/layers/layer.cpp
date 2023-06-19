#include "./fclayer.h"

namespace ai{
    BaseLayer::BaseLayer(alg::t_dim inp_size, alg::t_dim out_size):
        input_data{inp_size, out_size, arma::fill::none},
        input_size{inp_size},
        output_size{out_size}
        {}
    
    alg::t_mat BaseLayer::forward_propagation(alg::t_mat &im){
        // Save input
        input_data = im;
        // Return results
        return this->forward_propagation_implementation(im);
    }


    alg::t_mat BaseLayer::backward_propagation(alg::t_mat &out_error, alg::t_type alpha){
        // Update & Return results
        return this->backward_propagation_implementation(
            out_error,
            alpha
        );
    }

    std::ostream& operator<<(std::ostream& os, const BaseLayer& layer) {
        (&layer)->write(os); // Overload
        return os;
    }

    std::istream& operator>> (std::istream& is, BaseLayer&  layer) {
        (&layer)->read(is); // Overload
        return is;
    }
}



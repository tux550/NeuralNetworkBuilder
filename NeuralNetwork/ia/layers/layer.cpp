#include "./fclayer.h"

namespace ai{
    BaseLayer::BaseLayer(alg::t_dim inp_size, alg::t_dim out_size):
        input_data{},
        input_size{inp_size},
        output_size{out_size}
        {}
    
    alg::vec_mat BaseLayer::forward_propagation(alg::vec_mat &vec_im){
        input_data = vec_im;
        // Get results
        alg::vec_mat vec_res{};
        for (auto &m: vec_im) {
            vec_res.push_back(
                this->forward_propagation_implementation(m)
            );
        }
        // Return
        return vec_res;
    }


    alg::vec_mat BaseLayer::backward_propagation(alg::vec_mat &vec_out_error, alg::t_type alpha){
        // Get results
        alg::vec_mat vec_res{};
        for (auto i = 0; i < vec_out_error.size(); i++) {
            auto data = input_data[i];
            auto out_error = vec_out_error[i];
            vec_res.push_back(
                this->backward_propagation_implementation(data, out_error, alpha)
            );
        }
        // Return
        return vec_res;
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



#include <random>
#include <iostream>
#include "./fclayer.h"

namespace ai{
    // Constructor
    FCLayer::FCLayer(alg::t_dim inp_size, alg::t_dim out_size):
        BaseLayer{inp_size,out_size},
        weights_mat{inp_size, out_size}
        {
            // Generate random number generator
            alg::t_type lower_bound = 0;
            alg::t_type upper_bound = 1;
            std::uniform_real_distribution<alg::t_type> unif(lower_bound, upper_bound);
            std::default_random_engine re;
            // Generate random weights
            for (auto r=0; r<inp_size; r++) {
                for (auto c=0; c<out_size; c++) {
                    auto val = unif(re);
                    weights_mat.set_val(r,c,val);
                }
            }
        }
    // Forward Propagation
    alg::Matrix FCLayer::forward_propagation_implementation(alg::Matrix &im) {
        return alg::mat_prod(im, weights_mat);
    }
    // Backward Propagation
    alg::Matrix FCLayer::backward_propagation(alg::Matrix &out_error, alg::t_type alpha) {
        // TODO
        
        // Calc error
        auto in_error = alg::mat_prod(out_error, weights_mat.transpose());
        auto we_error = alg::mat_prod(input_data.transpose(), out_error);

        // Update
        /*
        std::cout << "UPDATE BY:";
        (we_error * alpha).display();
        std::cout << std::endl;
        */
        weights_mat = weights_mat - (we_error * alpha);
        // Return error
        return in_error;
    }
    // Setters
    void FCLayer::set_weights(alg::Matrix &w) {
        weights_mat = w;
    }
    // Getters
    alg::Matrix FCLayer::get_weights() {
        return weights_mat;
    }
}

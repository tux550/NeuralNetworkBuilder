#include <random>
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
    void FCLayer::forward_propagation() {
        output_mat = alg::mat_prod(input_mat, weights_mat);
    }
    // Backward Propagation
    void FCLayer::backward_propagation() {
        // TODO
    }
    // Setters
    void FCLayer::set_weights(alg::Matrix w) {
        weights_mat = w;
    }
}

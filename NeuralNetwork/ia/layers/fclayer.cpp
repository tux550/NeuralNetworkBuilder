#include "./fclayer.h"

namespace ai{
    // Constructor
    FCLayer::FCLayer(alg::t_dim inp_size, alg::t_dim out_size):
        BaseLayer{inp_size,out_size},
        weights_mat{inp_size, out_size, arma::fill::none},
        bias{1, out_size, arma::fill::none}
        {
            // Create random number generator
            alg::t_type lower_bound = -0.3;
            alg::t_type upper_bound = 0.3;
            std::uniform_real_distribution<alg::t_type> unif(lower_bound, upper_bound);
            std::default_random_engine re;
            // Generate random weights
            for (auto r=0; r<inp_size; r++) {
                for (auto c=0; c<out_size; c++) {
                    auto val = unif(re);
                    weights_mat(r,c)=val;
                }
            }
            // Generate bias
            for (auto c=0; c<out_size; c++) {
                auto val =0;//= unif(re);
                bias(0,c)=val;
            }
        }
    // Forward Propagation
    alg::t_mat FCLayer::forward_propagation_implementation(alg::t_mat &im) {
        //     mat_prod(im, w) + bias
        return (im * weights_mat) + bias;
    }
    // Backward Propagation
    alg::t_mat FCLayer::backward_propagation_implementation(alg::t_mat &out_error, alg::t_type alpha) {
        // Calc error

        alg::t_mat in_error = out_error * weights_mat.t();
        /*
        std::cout << "FCLAYER"<< std::endl
                 << "out error:" << arma::size(out_error) << std::endl
                 << "weights:" << arma::size(weights_mat) << std::endl
                 << "in error:" << arma::size(in_error) << std::endl;
        */
        alg::t_mat we_error = input_data.t() * out_error;

        // Update
        weights_mat = weights_mat - (we_error * alpha);
        bias        = bias - (out_error * alpha);

        // Return error
        return in_error;
    }
    // Setters
    void FCLayer::set_weights(alg::t_mat &w) {
        weights_mat = w;
    }
    // Getters
    alg::t_mat FCLayer::get_weights() {
        return weights_mat;
    }
    // Write
    void FCLayer::write(std::ostream& os) const {
        os << weights_mat << std::endl << bias << std::endl;
    }
    void FCLayer::read(std::istream& is) {
        alg::input_mat(is, weights_mat);
        alg::input_mat(is, bias);
    }
}

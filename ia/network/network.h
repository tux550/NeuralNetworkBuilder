#ifndef H_NETWORK
#define H_NETWORK

#include <memory>
#include <iostream>
#include <numeric>
#include <vector>
#include <random>
#include <algorithm>
#include "../../algebra/alg.h"
#include "../layers/layer.h"

namespace ai
{
    using ptr_layer  = std::shared_ptr<ai::BaseLayer>;
    using vec_layers = std::vector<ptr_layer>;
    using t_count    = int;

    class Network
    {
        private:
            vec_layers  layers;
            alg::t_mm2m loss;
            alg::t_mm2m loss_drv;
        public:
            // Constructor
            Network(alg::t_mm2m _loss, alg::t_mm2m _loss_drv);
            // Destructor
            ~Network();

            // Add layer
            void add_layer(ptr_layer layer);
            // Predict
            alg::vec_mat predict(alg::vec_mat &inp);
            // Fit
            void fit(alg::vec_mat &x_train, alg::vec_mat &y_train, t_count epochs, alg::t_type alpha, t_count batch_size=4, t_count epoch_intr = 1000);
    };

}


#endif
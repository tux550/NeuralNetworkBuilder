#ifndef H_NETWORK
#define H_NETWORK

#include <memory>
#include <iostream>
#include <numeric>
#include <vector>
#include <random>
#include <algorithm>
#include <string>
#include "../../algebra/alg.h"
#include "../layers/layer.h"

namespace ai
{
    using ptr_layer  = std::shared_ptr<ai::BaseLayer>;
    using vec_layers = std::vector<ptr_layer>;
    using t_count    = long;

    class Network
    {
        private:
            vec_layers  layers;
            alg::t_mm2t loss;
            alg::t_mm2m loss_drv;
        public:
            // Constructor
            Network(alg::t_mm2t _loss, alg::t_mm2m _loss_drv);
            static Network FullMLP(std::vector<alg::t_dim> vec_nodes_num, std::vector<alg::t_fmat> vec_act_func, std::vector<alg::t_fmat> vec_drv_func, alg::t_mm2t _loss, alg::t_mm2m _loss_drv);
            // Destructor
            ~Network();

            // Add layer
            void add_layer(ptr_layer layer);
            // Predict
            alg::vec_mat predict(alg::vec_mat &inp);
            alg::t_mat predict(alg::t_mat &inp);
            // Fit
            void fit(alg::vec_mat &x_train, alg::vec_mat &y_train, t_count epochs, alg::t_type alpha, t_count epoch_intr = 1);
            // Export
            void export_model(std::string out_filename);
            // Friend functions
            friend std::ostream& operator<<(std::ostream& os, const Network& nw);
            friend std::istream& operator>>(std::istream& is, Network& nw);
    };
}


#endif
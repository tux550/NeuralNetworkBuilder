#include "./network.h"
#include "../layers/actlayer.h"
#include "../layers/fclayer.h"
#include "../../logger/logger.h"


namespace ai
{
    // Constructor
    Network::Network(alg::t_mm2m _loss, alg::t_mm2m _loss_drv):
        layers{},
        loss{_loss},
        loss_drv{_loss_drv}
        {}

    Network Network::FullMLP(std::vector<alg::t_dim> vec_nodes_num, alg::t_t2t _act_func, alg::t_t2t _drv_func, alg::t_mm2m _loss, alg::t_mm2m _loss_drv) {
        auto nw = ai::Network(_loss, _loss_drv);

        for (auto i=0; i<vec_nodes_num.size() -1 ; i++) {
            auto nodes_in  = vec_nodes_num[i];
            auto nodes_out = vec_nodes_num[i+1];

            ai::ptr_layer fc_layer  = std::make_shared<ai::FCLayer>(nodes_in,nodes_out);
            ai::ptr_layer act_layer = std::make_shared<ai::ActLayer>(nodes_out, _act_func, _drv_func);

        nw.add_layer(fc_layer);
        nw.add_layer(act_layer);
        }
        return nw;
    }
    // Destructor
    Network::~Network() {}
    // Add layer
    void Network::add_layer(ptr_layer layer)
    {
        layers.push_back(layer);
    }
    // Predict 
    alg::vec_mat Network::predict(alg::vec_mat &inp) {
        alg::vec_mat res = inp;
        for (auto &l : layers) {
            res = l->forward_propagation(res);
        }
        return res;
    }
    // Fit
    void Network::fit(alg::vec_mat &x_train, alg::vec_mat &y_train, t_count epochs, alg::t_type alpha, t_count batch_size, t_count epoch_intr) {
        // Batch size
        if (batch_size > x_train.size()) {
            batch_size = x_train.size();
        }
        // Batch selection
        alg::vec_mat x_batch(batch_size);
        alg::vec_mat y_batch(batch_size);
        std::vector<std::size_t> batch_indexes(x_train.size());
        std::iota (std::begin(batch_indexes), std::end(batch_indexes), 0);

        // Random
        std::random_device rd;
        std::mt19937 gen(rd());

        // Epochs
        for (t_count i = 0; i < epochs; i++)
        {
            // Create batch
            std::shuffle(batch_indexes.begin(), batch_indexes.end(), gen);
            for (auto b=0; b<batch_size; b++) {
                x_batch[b] = x_train[batch_indexes[b]];
                y_batch[b] =  y_train[batch_indexes[b]];
            }

            alg::t_type error_total = 0;
            // Forward Propagation
            auto res = predict(x_batch);
            // Error
            alg::vec_mat error_vec;
            for (t_count j = 0; j < batch_size; j++) {
                error_total += loss(y_batch[j], res[j]).sum();
                auto error = loss_drv(y_batch[j], res[j]);
                error_vec.push_back(error);
            }


            // Backward Propagation
            for (int k = layers.size()-1; k>= 0; k--){
                //std::cout << "LAYER:" << k  << std::endl;
                error_vec = layers[k]->backward_propagation(error_vec, alpha);
            } 
            if (i%epoch_intr == 0) {
                verbose_print("Epoch " + std::to_string(i) + "/" + std::to_string(epochs) + " Error=" + std::to_string(error_total));
            }

        }
    }
}

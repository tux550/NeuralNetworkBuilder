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

    Network Network::FullMLP(std::vector<alg::t_dim> vec_nodes_num, std::vector<alg::t_fmat> vec_act_func, std::vector<alg::t_fmat> vec_drv_func, alg::t_mm2m _loss, alg::t_mm2m _loss_drv) {
        auto nw = ai::Network(_loss, _loss_drv);

        for (auto i=0; i<vec_nodes_num.size() -1 ; i++) {
            auto nodes_in  = vec_nodes_num[i];
            auto nodes_out = vec_nodes_num[i+1];

            ai::ptr_layer fc_layer  = std::make_shared<ai::FCLayer>(nodes_in,nodes_out);
            ai::ptr_layer act_layer = std::make_shared<ai::ActLayer>(nodes_out, vec_act_func[i], vec_drv_func[i]);

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
        alg::vec_mat res;
        for (auto X : inp){
            for (auto &l : layers) {
                X = l->forward_propagation(X);
            }
            res.push_back(X);
        }
        return res;
    }
    alg::t_mat Network::predict(alg::t_mat &inp) {
        alg::t_mat X = inp;
        for (auto &l : layers) {
            X = l->forward_propagation(X);
        }
        return X;
    }

    // Fit
    void Network::fit(alg::vec_mat &x_train, alg::vec_mat &y_train, t_count epochs, alg::t_type alpha, t_count epoch_intr) {
        // Order selection selection
        std::vector<std::size_t> input_indexes(x_train.size());
        std::iota (std::begin(input_indexes), std::end(input_indexes), 0);

        // Random
        std::random_device rd;
        std::mt19937 gen(rd());

        // Epochs
        for (t_count i = 0; i < epochs; i++)
        {   
            // Epoch total error
            alg::t_type total_error;
            alg::t_mat error;

            // Reshuffle
            std::shuffle(input_indexes.begin(), input_indexes.end(), gen);
            // Error for each element
            for (auto &idx : input_indexes) {
                // Forward Propagation
                auto res = predict(x_train[idx]);
                // Error
                total_error += arma::accu(loss(y_train[idx], res));
                error = loss_drv(y_train[idx], res);
                // Backward Propagation
                for (int k = layers.size()-1; k>= 0; k--){
                    //std::cout << "K:" << k << " shape:" << arma::size(error) << std::endl;
                    error = layers[k]->backward_propagation(error, alpha);
                } 
            }            


            //if ( (i%epoch_intr == 0) || (i+1==epochs) ) {
            verbose_print("Epoch " + std::to_string(i) + "/" + std::to_string(epochs) + " Error=" + std::to_string(total_error));
            //}

        }
    }
    // Export
    void Network::export_model(std::string out_filename) {
        // TODO
        return;
    }

    std::ostream& operator<<(std::ostream& os, const Network& nw) {
        for (auto &l: nw.layers) {
            os << (*l) << std::endl;
        }
        return os;
    }

    std::istream& operator>>(std::istream& is, Network& nw) {
        for (auto &l: nw.layers) {
            is >> (*l);
        }
        return is;
    }
}

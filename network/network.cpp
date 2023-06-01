#include "./network.h"

namespace ai
{
    // Constructor
    Network::Network(alg::t_mm2t _loss, alg::t_mm2t _loss_drv):
        layers{},
        loss{_loss},
        loss_drv{_loss_drv}
        {}
    // Destructor
    Network::~Network() {}
    // Add layer
    void Network::add_layer(ptr_layer layer)
    {
        layers.push_back(layer);
    }
    // Predict 
    alg::Matrix Network::predict(alg::Matrix &inp) {
        alg::Matrix res = inp;
        for (auto &l : layers) {
            res = l->forward_propagation(res);
        }
        return res;
    }
}

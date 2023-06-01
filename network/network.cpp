#include "./network.h"

namespace ai
{
    // Constructor
    Network::Network(): layers{} {}
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

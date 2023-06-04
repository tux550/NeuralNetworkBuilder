
#include <iostream>
#include "./network.h"

namespace ai
{
    // Constructor
    Network::Network(alg::t_mm2m _loss, alg::t_mm2m _loss_drv):
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
    alg::MultidimMatrix Network::predict(alg::MultidimMatrix &inp) {
        alg::MultidimMatrix res = inp;
        for (auto &l : layers) {
            res = l->forward_propagation(res);
        }
        return res;
    }
    // Fit
    void Network::fit(vec_mat &x_train, vec_mat &y_train, t_count epochs, alg::t_type alpha, t_count epoch_intr) {
        // Epochs
        for (t_count i = 0; i < epochs; i++)
        {
            alg::t_type error_total = 0;
            for (t_count j = 0; j < x_train.size(); j++) {
                // Forward Propagation
                //std::cout << "FORWARD PROPAGATION" << std::endl;
                auto res = predict(x_train[j]);
                // Error
                //std::cout << "ERROR" << std::endl;
                error_total += loss(y_train[j], res).sum();
                // Backward Propagation
                //std::cout << "LOSS DRV" << std::endl;
                //res.display();
                auto error = loss_drv(y_train[j], res);
                //error.display();
                //std::cout << "BACKWARD PROPAGATION" << std::endl;
                for (int k = layers.size()-1; k>= 0; k--){
                    //std::cout << "LAYER:" << k  << std::endl;
                    error = layers[k]->backward_propagation(error, alpha);
                } 
            }
            if (i%epoch_intr == 0) {
                std::cout << "Epoch: " << i << std::endl;
                std::cout << "Error: " << error_total << std::endl;
            }
        }
        
    }
    /*
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))
    */
}

#ifndef H_NETWORK
#define H_NETWORK

#include <memory>
#include "../algebra/alg.h"
#include "../layers/layer.h"

namespace ai
{
    using ptr_layer  = std::shared_ptr<BaseLayer>;
    using vec_layers = std::vector<ptr_layer>;
    using vec_mat    = std::vector<alg::Matrix>;
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
            alg::Matrix predict(alg::Matrix &inp);
            // Fit
            void fit(vec_mat &x_train, vec_mat &y_train, t_count epochs, alg::t_type alpha);
    };

}


#endif
#ifndef H_NETWORK
#define H_NETWORK

#include <memory>
#include "../algebra/alg.h"
#include "../layers/layer.h"

namespace ai
{
    using ptr_layer  = std::shared_ptr<BaseLayer>;
    using vec_layers = std::vector<ptr_layer>;

    class Network
    {
        private:
            vec_layers  layers;
            alg::t_mm2t loss;
            alg::t_mm2t loss_drv;
        public:
            // Constructor
            Network(alg::t_mm2t _loss, alg::t_mm2t _loss_drv);
            // Destructor
            ~Network();

            // Add layer
            void add_layer(ptr_layer layer);
            // Predict
            alg::Matrix predict(alg::Matrix &inp);
    };

}


#endif
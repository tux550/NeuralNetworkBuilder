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
            vec_layers layers;
        public:
            // Constructor
            Network();
            // Destructor
            ~Network();

            // Add layer
            void add_layer(ptr_layer layer);
            // Predict
            alg::Matrix predict(alg::Matrix &inp);
    };

}


#endif
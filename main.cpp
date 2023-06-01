#include <iostream>
#include <vector>
#include <memory>
#include "algebra/alg.h"
#include "layers/actlayer.h"
#include "layers/fclayer.h"
#include "network/network.h"
#include "funcs/functions.h"
using namespace std;

int main() {

    auto a = alg::Matrix( alg::t_mat{ {1,2,3}} );
    auto b = alg::Matrix( alg::t_mat{{-5,2},{-4,4},{-3,1}});
    auto c = alg::Matrix( alg::t_mat{{1},{2}});
    cout << "INPUT" << endl;
    a.display();
    cout << "WEIGHTS" << endl;
    b.display();

    // Create Network
    auto nw = ai::Network();

    // Add layers
    std::shared_ptr<ai::FCLayer> fc1 ( new ai::FCLayer(3.0,2.0) );
    std::shared_ptr<ai::ActLayer> act1 ( new ai::ActLayer(3.0, relu, relu_drv) );
    std::shared_ptr<ai::FCLayer> fc2 ( new ai::FCLayer(2.0,1.0) );
    fc1->set_weights(b);
    fc2->set_weights(c);
    nw.add_layer(fc1);
    nw.add_layer(act1);
    nw.add_layer(fc2);

    // Predict
    auto r = nw.predict(a);
    cout << "RESULTS" << endl;
    r.display();
    return 0;
}
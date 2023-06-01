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
    ai::vec_mat x_train = { 
        alg::Matrix( alg::t_mat{ {1,1}} ),
        alg::Matrix( alg::t_mat{ {0,0}} ),
        alg::Matrix( alg::t_mat{ {0,1}} ),
        alg::Matrix( alg::t_mat{ {1,0}} ),
    };

    ai::vec_mat y_train = {
        alg::Matrix( alg::t_mat{ {0}} ),
        alg::Matrix( alg::t_mat{ {0}} ),
        alg::Matrix( alg::t_mat{ {1}} ),
        alg::Matrix( alg::t_mat{ {1}} ),
    };

    std::shared_ptr<ai::BaseLayer> fc1 = std::make_shared<ai::FCLayer>(2.0,3.0);
    std::shared_ptr<ai::BaseLayer> act1 = std::make_shared<ai::ActLayer>(3.0, hypertan, hypertan_drv);
    //std::shared_ptr<ai::BaseLayer> act1 = std::make_shared<ai::ActLayer>(3.0, relu, relu_drv);
    std::shared_ptr<ai::BaseLayer> fc2 = std::make_shared<ai::FCLayer>(3.0,1.0);
    std::shared_ptr<ai::BaseLayer> act2 = std::make_shared<ai::ActLayer>(1.0, hypertan, hypertan_drv);
    //std::shared_ptr<ai::BaseLayer> act2 = std::make_shared<ai::ActLayer>(1.0, relu, relu_drv);

    auto nw = ai::Network(mse, mse_drv);
    nw.add_layer(fc1);
    nw.add_layer(act1);
    nw.add_layer(fc2);
    nw.add_layer(act2);


    std::dynamic_pointer_cast<ai::FCLayer> (fc1) -> get_weights().display();
    nw.predict( x_train[0] ).display();
    nw.fit(x_train,y_train,1000,0.1);
    std::dynamic_pointer_cast<ai::FCLayer> (fc1) -> get_weights().display();
    nw.predict( x_train[0] ).display();
    nw.predict( x_train[1] ).display();
    nw.predict( x_train[2] ).display();
    nw.predict( x_train[3] ).display();

    /*
    auto a = alg::Matrix( alg::t_mat{ {1,2,3}} );
    auto b = alg::Matrix( alg::t_mat{{-5,2},{-4,4},{-3,1}});
    auto c = alg::Matrix( alg::t_mat{{1},{2}});
    cout << "INPUT" << endl;
    a.display();
    cout << "WEIGHTS" << endl;
    b.display();

    // Create Network
    auto nw = ai::Network(mse, mse_drv);

    // Add layers
    std::shared_ptr<ai::BaseLayer> fc1 = std::make_shared<ai::FCLayer>(3.0,2.0);
    std::shared_ptr<ai::BaseLayer> act1 = std::make_shared<ai::ActLayer>(3.0, relu, relu_drv);
    std::shared_ptr<ai::BaseLayer> fc2 = std::make_shared<ai::FCLayer>(2.0,1.0);
    std::dynamic_pointer_cast<ai::FCLayer> (fc1) ->set_weights(b);
    std::dynamic_pointer_cast<ai::FCLayer> (fc2) ->set_weights(c);
    nw.add_layer(fc1);
    nw.add_layer(act1);
    nw.add_layer(fc2);

    // Predict
    auto r = nw.predict(a);
    cout << "RESULTS" << endl;
    r.display();
    return 0;
    */
}
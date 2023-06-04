#include <iostream>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream> 
#include <string>
#include <sstream>
#include "algebra/alg.h"
#include "layers/actlayer.h"
#include "layers/fclayer.h"
#include "network/network.h"
#include "funcs/functions.h"


using namespace std;

int get_maximum_index(alg::MultidimMatrix& y_pred) {
    int res = 0;
    auto val = y_pred.get_val({0,0});
    for (alg::t_dim i=1; i<y_pred.get_shape()[1]; i++) {
        auto new_val = y_pred.get_val({0,i});
        if (new_val > val) {
            res = i;
            val = new_val;
        }
    }
    return res;
}

int main() {
    /*
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
    */


    cout << "Init Layers" << endl;
    ai::ptr_layer fc1  = std::make_shared<ai::FCLayer>(4.0,8.0);
    ai::ptr_layer act1 = std::make_shared<ai::ActLayer>(8.0, hypertan, hypertan_drv); //relu, relu_drv);
    ai::ptr_layer fc2  = std::make_shared<ai::FCLayer>(8.0,5.0);
    ai::ptr_layer act2 = std::make_shared<ai::ActLayer>(5.0, hypertan, hypertan_drv); // relu, relu_drv);
    ai::ptr_layer fc3  = std::make_shared<ai::FCLayer>(5.0,3.0);
    ai::ptr_layer act3 = std::make_shared<ai::ActLayer>(3.0, hypertan, hypertan_drv); // relu, relu_drv);
    cout << "End Layers" << endl;

    
    auto nw = ai::Network(mse, mse_drv);
    nw.add_layer(fc1);
    nw.add_layer(act1);
    nw.add_layer(fc2);
    nw.add_layer(act2);
    nw.add_layer(fc3);
    nw.add_layer(act3);

    ifstream XFile("datasets/x.csv");
    ifstream YFile("datasets/y.csv");

    string line;
    string num;
    char delim = ' '; 

    ai::vec_mat x_train{};
    while (std::getline(XFile, line)) {
        std::stringstream ss(line);
        alg::t_mat2d tmp_mat{};
        alg::t_mat1d tmp_row{};
        while (std::getline(ss, num, delim)) {
            tmp_row.push_back( std::stod(num) );
        }
        tmp_mat.push_back(tmp_row);
        x_train.push_back(alg::MultidimMatrix::FromMat2D(tmp_mat));
    }

    ai::vec_mat y_train{};
    while (std::getline(YFile, line)) {
        std::stringstream ss(line);
        alg::t_mat2d tmp_mat{};
        alg::t_mat1d tmp_row{};
        while (std::getline(ss, num, delim)) {
            tmp_row.push_back( std::stod(num) );
        }
        tmp_mat.push_back(tmp_row);
        y_train.push_back(alg::MultidimMatrix::FromMat2D(tmp_mat));
    }


    std::cout << "INIT WEIGHTS OF L0" << endl;
    auto w0 = std::dynamic_pointer_cast<ai::FCLayer> (fc1) -> get_weights();
    std::cout << "DISPLAY WEIGHTS OF L0" << endl;
    w0.display();
    std::cout << "PREDICT" << std::endl;
    auto p = nw.predict( x_train[0] );
    std::cout << "END PREDICT" << std::endl;
    p.display();
    std::cout << "TRAIN" << endl;
    nw.fit(x_train,y_train,5000,0.01, 1);
    std::cout << "FINAL WEIGHTS OF L0" << endl;
    std::dynamic_pointer_cast<ai::FCLayer> (fc1) -> 
    get_weights().display();

    vector<double> misses(3);
    vector<double> total(3);
    for (int ind = 0; ind < x_train.size(); ind ++) {
        auto tmp = nw.predict(x_train[ind]);
        auto real = get_maximum_index( y_train[ind] );
        if ( real != get_maximum_index( tmp ) ) {
            misses[real] += 1;            
        }
        total[real] += 1;
    }
    cout << "Misses:" << endl;
    for (int i=0; i<3; i++) {
        cout << misses[i] / total[i] <<  std::endl;
    } 

    
    //nw.predict(X).display();
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
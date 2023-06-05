#include <iostream>
#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <stdexcept>
#include "load/load.h"
#include "algebra/alg.h"
#include "ia/layers/actlayer.h"
#include "ia/layers/fclayer.h"
#include "ia/network/network.h"
#include "funcs/functions.h"


using namespace std;
int get_prediction(alg::Matrix& y_pred);

ai::Network nw_from_inputs( ) {
    // Init
    alg::t_dim in_size, out_size;
    alg::t_dim depth, nodes;
    string str_act_func, str_loss_func;
    // Input
    std::cin >>in_size >> out_size >> depth >> nodes >> str_act_func >> str_loss_func;
    // Vector
    std::vector<alg::t_dim> layers_vector;
    layers_vector.push_back(in_size);
    for (int i=0;i<depth;i++) {layers_vector.push_back(nodes);}
    layers_vector.push_back(out_size);
    (depth, nodes);
    // Cases
    alg::t_t2t act_func, act_drv;
    if (str_act_func == "hypertan") {
        act_func = hypertan;
        act_drv = hypertan_drv;
    } else {
        throw std::invalid_argument("Invalid act func");
    }
    alg::t_mm2m loss_func, loss_drv;
    if (str_loss_func == "mse") {
        loss_func = mse;
        loss_drv = mse_drv;
    } else {
        throw std::invalid_argument("Invalid loss func");
    }
    // Create
    return ai::Network::FullMLP(layers_vector,act_func, act_drv, loss_func, loss_drv);;
}

int main() {
    std::cout << "Create Network" << std::endl;
    auto nw = nw_from_inputs();
    //auto nw = ai::Network::FullMLP({4,20,20,3},hypertan, hypertan_drv,mse, mse_drv);

    std::cout << "Load Dataset" << std::endl;
    auto x_train = load_file("datasets/x.csv");
    auto y_train = load_file("datasets/y.csv");

    std::cout << "Train" << endl;
    nw.fit(x_train,y_train,10000,0.01);
    
    std::cout << "Result" << endl;
    auto res = nw.predict(x_train);


    std::cout << "Stadistics" << endl;
    vector<double> misses(3);
    vector<double> total(3);
    
    for (int ind = 0; ind < res.size(); ind ++) {
        auto real = get_prediction( y_train[ind] );
        if ( real != get_prediction( res[ind] ) ) {
            misses[real] += 1;            
        }
        total[real] += 1;
    }
    cout << "Misses:" << endl;
    for (int i=0; i<3; i++) {
        cout << misses[i] / total[i] <<  std::endl;
    } 
    

}










// TEMPORARY:
int get_prediction(alg::Matrix& y_pred) {
    int res = 0;
    auto val = y_pred.get_val(0,0);
    for (int i=1; i<y_pred.get_cols(); i++) {
        auto new_val = y_pred.get_val(0,i);
        if (new_val > val) {
            res = i;
            val = new_val;
        }
    }
    return res;
}
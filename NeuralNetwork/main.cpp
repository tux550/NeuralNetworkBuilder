#include <iostream>
#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>
#include "load/load.h"
#include "algebra/alg.h"
#include "ia/layers/actlayer.h"
#include "ia/layers/fclayer.h"
#include "ia/network/network.h"
#include "funcs/functions.h"
#include "logger/logger.h"


using namespace std;
int get_prediction(alg::Matrix& y_pred);

void mode_from_inputs(int &train_load, int &test,int &save) {
    std::cin >> train_load >> test >> save;
}

ai::Network nw_from_inputs( ) {
    // Init
    alg::t_dim depth;
    std::vector<alg::t_dim> vec_nodes;
    std::vector<alg::t_t2t> vec_act_func;
    std::vector<alg::t_t2t> vec_act_drv;
    alg::t_mm2m loss_func, loss_drv;
    // Tmp
    alg::t_dim nodes;
    std::string str_act_func, str_loss_func;
    // Input
    std::cin >> depth;
    // Input for each layer
    for (auto n=0; n< depth;n++) {
        std::cin >> nodes;

        vec_nodes.push_back(nodes);

        if (n != 0) {
            std::cin >>  str_act_func;
            if (str_act_func == "hypertan") {
                vec_act_func.push_back(hypertan);
                vec_act_drv.push_back(hypertan_drv);
            } else if (str_act_func == "relu") {
                vec_act_func.push_back(relu);
                vec_act_drv.push_back(relu_drv);
            } else {
                std::cout << str_act_func << " " << n << std::endl;
                throw std::invalid_argument("Invalid act func");
            }
        }       
    }
    
    std::cin >> str_loss_func;
    if (str_loss_func == "mse") {
        loss_func = mse;
        loss_drv = mse_drv;
    } else {
        throw std::invalid_argument("Invalid loss func");
    }
    
    // Create
    return ai::Network::FullMLP(vec_nodes,vec_act_func,vec_act_drv, loss_func, loss_drv);;
}

alg::vec_mat dataset_from_inputs() {
    std::string filename;
    std::cin  >> filename;
    return load_file(filename);
}

void train_from_inputs(ai::Network& nw, alg::vec_mat& x_train, alg::vec_mat& y_train) {
    // Init
    ai::t_count epochs, batch_size;
    alg::t_type alpha; 
    // Get data
    std::cin >> epochs >> alpha >> batch_size;
    // Train
    nw.fit(x_train,y_train,epochs, alpha, batch_size);
}


void load_from_inputs(ai::Network& nw){
    // Init
    std::string nw_filename; 
    // Get data
    std::cin >> nw_filename;
    // Load from file
    ifstream NwFile(nw_filename);
    NwFile >> nw;
}

void save_from_inputs(ai::Network& nw){
    // Init
    std::string nw_filename; 
    // Get data
    std::cin >> nw_filename;
    // Load from file
    ofstream NwFile(nw_filename);
    NwFile << nw;
}

int main() {
    int train_load, test, save;
    debug_print("Mode");
    mode_from_inputs(train_load, test, save);   

    debug_print("Create Network Architecture");
    auto nw = nw_from_inputs(); //auto nw = ai::Network::FullMLP({4,20,20,3},hypertan, hypertan_drv,mse, mse_drv);
    auto nw2 = nw;

    if (train_load == 1) 
    {
        // TRAIN MODE
        debug_print("Load Train Dataset");
        auto x_train = dataset_from_inputs(); //load_file("../Dataset/x.csv");
        auto y_train = dataset_from_inputs(); //load_file("../Dataset/y.csv");
        debug_print("Train");
        train_from_inputs(nw, x_train, y_train); // nw.fit(x_train,y_train,10000,0.01);
    }
    else {
        // LOAD MDOE
        load_from_inputs(nw);
    }

    if (test == 1) // TEST MODE
    {
        debug_print("Load Test Dataset");
        auto x_test = dataset_from_inputs();

        debug_print("Result");
        auto res = nw.predict(x_test);

        debug_print("Output");
        for (auto &m : res){
            std::cout << m;
        }
    }

    if (save == 1) // SAVE MODE
    {
        debug_print("Saving nw");
        save_from_inputs(nw);
    }



    


    /*
    stringstream buffer; 
    buffer << nw;
    buffer >> nw2;
    debug_print("nw2");
        std::cout << nw2;
    */
    /*
    debug_print("Stadistics");
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
    */

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

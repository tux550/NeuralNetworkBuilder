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

void mode_from_inputs(int &train_load, int &test,int &save) {
    std::cin >> train_load >> test >> save;
}

ai::Network nw_from_inputs( ) {
    // Init
    alg::t_dim depth;
    std::vector<alg::t_dim> vec_nodes;
    std::vector<alg::t_fmat> vec_act_func;
    std::vector<alg::t_fmat> vec_act_drv;
    alg::t_mm2t loss_func;
    alg::t_mm2m loss_drv;
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
            } else if (str_act_func == "sigmoid")  {
                vec_act_func.push_back(sigmoid);
                vec_act_drv.push_back(sigmoid_drv);
            }
            else {
                std::cout << str_act_func << " " << n << std::endl;
                throw std::invalid_argument("Invalid act func");
            }
        }       
    }
    
    std::cin >> str_loss_func;
    if (str_loss_func == "mse") {
        loss_func = mse;
        loss_drv = mse_drv;
    }else if (str_loss_func == "cross_entropy") {
        loss_func = cross_entropy;
        loss_drv = cross_entropy_drv;
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
    ai::t_count epochs;
    alg::t_type alpha; 
    // Get data
    std::cin >> epochs >> alpha;
    // Train
    nw.fit(x_train,y_train,epochs, alpha);
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
    auto nw = nw_from_inputs();
    auto nw2 = nw;

    if (train_load == 1) 
    {
        // TRAIN MODE
        debug_print("Load Train Dataset");
        auto x_train = dataset_from_inputs(); 
        auto y_train = dataset_from_inputs();
        debug_print("Train");
        train_from_inputs(nw, x_train, y_train);
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

}







#include "./functions.h"
#include <iostream>

alg::t_type s(alg::t_type x) {
    return 1 / (1+std::exp(-x));
}

void relu(alg::t_type &x){
    if (x<=0) {x = 0;}
}

void relu_drv(alg::t_type &x){
    if (x>0) {x = 1;}
    else {x = 0;}
}

void hypertan(alg::t_type &x){
    x = std::tanh(x);
}

void hypertan_drv(alg::t_type &x){
    x = 1-std::pow(std::tanh(x),2);
}

void sigmoid(alg::t_type &x){
    x = s(x);
}

void sigmoid_drv(alg::t_type &x){
    x = s(x) * (1-s(x));
}


alg::t_mat mse(alg::t_mat &y_true, alg::t_mat &y_pred) {
    // MSE: SUM{(y-y')**2} * (1/n)
    alg::t_type invn = (1.0/arma::size(y_true)[1]);
    return arma::pow((y_pred-y_true),2.0) * invn;
    //return  arma::accu(arma::pow((a-b),2)) / arma::size(a)[1];
}

alg::t_mat mse_drv(alg::t_mat &y_true, alg::t_mat &y_pred) {
    // drvMSE: SUM{(y-y')} * (2/n)
    alg::t_mat dif = y_pred-y_true;
    alg::t_type invn = (1.0/arma::size(y_true)[1]);
    return dif.for_each([](alg::t_type& x) { x = x*2.0;}) * invn ;
    //return arma::accu(a-b) * (2/arma::size(a)[1]);
}
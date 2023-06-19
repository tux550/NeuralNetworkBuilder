#include "./functions.h"
#include <iostream>

alg::t_type s(alg::t_type x) {
    return 1 / (1+std::exp(-x));
}

alg::t_mat softmax(alg::t_mat y) {
    y.for_each([](alg::t_type& val) { val = std::exp(val);});
    alg::t_type all_sum = arma::accu(y);
    return y/all_sum;
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


alg::t_type mse(alg::t_mat &y_true, alg::t_mat &y_pred) {
    // MSE: SUM{(y-y')**2} * (1/n)
    alg::t_type invn = (1.0/arma::size(y_true)[1]);
    return arma::accu( arma::pow((y_pred-y_true),2.0) * invn );
}

alg::t_mat mse_drv(alg::t_mat &y_true, alg::t_mat &y_pred) {
    // drvMSE: SUM{(y-y')} * (2/n)
    alg::t_mat dif = y_pred-y_true;
    alg::t_type invn = (1.0/arma::size(y_true)[1]);
    return dif.for_each([](alg::t_type& x) { x = x*2.0;}) * invn ;

}


alg::t_type cross_entropy(alg::t_mat &y_true, alg::t_mat &y_pred) {
    // CrossEntropy: SUM{-y*log(y')}
    alg::t_mat y_soft_pred = softmax(y_pred);
    return (-1.0)*arma::accu(y_true % arma::log(y_soft_pred));
}

alg::t_mat cross_entropy_drv(alg::t_mat &y_true, alg::t_mat &y_pred) {
    // drvCrossEntropy: SUM{y'-y}
    return y_pred-y_true;
}
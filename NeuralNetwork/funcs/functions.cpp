#include "./functions.h"

alg::t_mat relu(alg::t_mat &X){
    alg::t_mat res = X;
    res.for_each( [](alg::t_type& val) {
        if (val<=0) {val = 0;}
    });
    return res;
}

alg::t_mat relu_drv(alg::t_mat &X){
    alg::t_mat res = X;
    res.for_each( [](alg::t_type& val) {
        if (val>0) {val = 1;}
        else {val = 0;}
    });
    return res;
}

alg::t_mat hypertan(alg::t_mat &X){
    alg::t_mat res = X;
    res.for_each( [](alg::t_type& val) {
        val = std::tanh(val);
    });
    return res;
}

alg::t_mat hypertan_drv(alg::t_mat &X){
    alg::t_mat res = X;
    res.for_each( [](alg::t_type& val) {
        val = 1-std::pow(std::tanh(val),2);
    });
    return res;
}

alg::t_mat mse(alg::t_mat &y_true, alg::t_mat &y_pred) {
    // MSE: SUM{(y-y')**2} * (1/n)
    return arma::pow((y_pred-y_true),2);
    //return  arma::accu(arma::pow((a-b),2)) / arma::size(a)[1];
}

alg::t_mat mse_drv(alg::t_mat &y_true, alg::t_mat &y_pred) {
    // drvMSE: SUM{(y-y')} * (2/n)
    alg::t_mat dif = y_pred-y_true;
    return dif.for_each([](alg::t_type& x) { x = x*2;});
    //return arma::accu(a-b) * (2/arma::size(a)[1]);
}
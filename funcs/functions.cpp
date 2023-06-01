#include "./functions.h"

alg::t_type relu(alg::t_type x){
    if (x>0) {return x;}
    else {return 0;}
}

alg::t_type relu_drv(alg::t_type x){
    if (x>0) {return 1;}
    else {return 0;}
}

alg::t_type hypertan(alg::t_type x){
    return std::tanh(x);
}

alg::t_type hypertan_drv(alg::t_type x){
    return 1-std::pow(std::tanh(x),2);
}

alg::Matrix mse(alg::Matrix &a, alg::Matrix &b) {
    // Validate
    if (a.get_rows() != b.get_rows()) {
        throw std::invalid_argument("mse: rows missmatch");
    }
    if (a.get_cols() != b.get_cols()) {
        throw std::invalid_argument("mse: cols missmatch");
    }
    // MSE
    auto mse = alg::Matrix(1, a.get_cols());
    alg::t_type n = a.get_cols();
    for (auto c = 0; c < a.get_cols(); c++)
    {
        alg::t_type tmp = 0;
        for (auto r = 0; r < a.get_rows(); r++)
        {
            // (y_true-y_pred)**2
            tmp += std::pow((a.get_val(r,c)-b.get_val(r,c)),2);
        }
        mse.set_val(0,c,tmp);
    }
    mse = mse * (1/n); // 1/n * SUM{(y-y')**2} 
    // Return
    return mse;
}

alg::Matrix mse_drv(alg::Matrix &a, alg::Matrix &b) {
    // Validate
    if (a.get_rows() != b.get_rows()) {
        throw std::invalid_argument("mse_drv: rows missmatch");
    }
    if (a.get_cols() != b.get_cols()) {
        throw std::invalid_argument("mse_drv: cols missmatch");
    }
    // MSE DRV
    auto mse_drv = alg::Matrix(1, a.get_rows());
    alg::t_type n = a.get_cols();
    for (auto c = 0; c < a.get_cols(); c++)
    {
        alg::t_type tmp = 0;
        for (auto r = 0; r < a.get_rows(); r++)
        {
            // (y_pred-y_true)
            tmp += (b.get_val(r,c)-a.get_val(r,c));
        }
        mse_drv.set_val(0,c,tmp);
    }
    mse_drv = mse_drv * (2/n); // 2/n * SUM{(y-y')}
    // Return
    return mse_drv;
}
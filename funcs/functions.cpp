#include "./functions.h"

alg::t_type relu(alg::t_type x){
    if (x>0) {return x;}
    else {return 0;}
}

alg::t_type relu_drv(alg::t_type x){
    if (x>0) {return 1;}
    else {return 0;}
}

alg::t_type mse(alg::Matrix &a, alg::Matrix &b) {
    // Validate
    if (a.get_rows() != b.get_rows()) {
        throw std::invalid_argument("mse: rows missmatch");
    }
    if (a.get_cols() != b.get_cols()) {
        throw std::invalid_argument("mse: cols missmatch");
    }
    // MSE
    alg::t_type mse = 0;
    alg::t_type n = a.get_rows()* a.get_cols();
    for (auto r = 0; r < a.get_rows(); r++)
    {
        for (auto c = 0; c < a.get_cols(); c++)
        {
            // (y-y')**2
            mse += std::pow((a.get_val(r,c)-b.get_val(r,c)),2);
        }
    }
    mse = mse/n; // 1/n * SUM{(y-y')**2} 
    // Return
    return mse;
}

alg::t_type mse_drv(alg::Matrix &a, alg::Matrix &b) {
    // Validate
    if (a.get_rows() != b.get_rows()) {
        throw std::invalid_argument("mse_drv: rows missmatch");
    }
    if (a.get_cols() != b.get_cols()) {
        throw std::invalid_argument("mse_drv: cols missmatch");
    }
    // MSE drv
    alg::t_type mse_drv = 0;
    alg::t_type n = a.get_rows()* a.get_cols();
    for (auto r = 0; r < a.get_rows(); r++)
    {
        for (auto c = 0; c < a.get_cols(); c++)
        {
            // (y-y')
            mse_drv += (a.get_val(r,c)-b.get_val(r,c));
        }
    }
    mse_drv = 2*mse_drv/n; // 2/n * SUM{(y-y')}
    // Return
    return mse_drv;
}
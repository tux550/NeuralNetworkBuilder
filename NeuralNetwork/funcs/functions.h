#ifndef H_FUNCS
#define H_FUNCS

#include <cmath>
#include "../algebra/alg.h"

alg::t_type relu(alg::t_type x);
alg::t_type relu_drv(alg::t_type x);
alg::t_type hypertan(alg::t_type x);
alg::t_type hypertan_drv(alg::t_type x);
alg::Matrix mse(alg::Matrix &a, alg::Matrix &b);
alg::Matrix mse_drv(alg::Matrix &a, alg::Matrix &b);

class ActivationFunctor {
    alg::t_t2t act_fun;
    alg::t_t2t act_drv;
};

class LossFunctor {
    alg::t_mm2m loss_fun;
    alg::t_mm2m loss_drv;
};

#endif
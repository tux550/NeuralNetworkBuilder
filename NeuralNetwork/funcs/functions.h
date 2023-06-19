#ifndef H_FUNCS
#define H_FUNCS

#include <algorithm>
#include <cmath>
#include "../algebra/alg.h"

alg::t_mat relu(alg::t_mat &X);
alg::t_mat relu_drv(alg::t_mat &X);
alg::t_mat hypertan(alg::t_mat &X);
alg::t_mat hypertan_drv(alg::t_mat &X);
alg::t_mat mse(alg::t_mat &a, alg::t_mat &b);
alg::t_mat mse_drv(alg::t_mat &a, alg::t_mat &b);

#endif
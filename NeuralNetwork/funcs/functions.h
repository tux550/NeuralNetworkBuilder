#ifndef H_FUNCS
#define H_FUNCS

#include <algorithm>
#include <cmath>
#include "../algebra/alg.h"

void relu(alg::t_type &x);
void relu_drv(alg::t_type &x);
void hypertan(alg::t_type &x);
void hypertan_drv(alg::t_type &x);
alg::t_mat mse(alg::t_mat &a, alg::t_mat &b);
alg::t_mat mse_drv(alg::t_mat &a, alg::t_mat &b);

#endif
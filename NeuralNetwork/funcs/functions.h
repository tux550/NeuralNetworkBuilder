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


#endif
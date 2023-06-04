#ifndef H_FUNCS
#define H_FUNCS

#include <cmath>
#include "../algebra/alg.h"

alg::t_type relu(alg::t_type x);
alg::t_type relu_drv(alg::t_type x);
alg::t_type hypertan(alg::t_type x);
alg::t_type hypertan_drv(alg::t_type x);
alg::MultidimMatrix mse(alg::MultidimMatrix &a, alg::MultidimMatrix &b);
alg::MultidimMatrix mse_drv(alg::MultidimMatrix &a, alg::MultidimMatrix &b);


#endif
#include "./functions.h"

alg::t_type relu(alg::t_type x){
    if (x>0) {return x;}
    else {return 0;}
}

alg::t_type relu_drv(alg::t_type x){
    if (x>0) {return 1;}
    else {return 0;}
}
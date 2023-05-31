#include <iostream>
#include <vector>
#include "alg.h"
using namespace std;

int main() {

    auto a = alg::Matrix( {{2,3}, {3,4}, {7,7}} );
    auto b = alg::Matrix( {{5,2},{4,4}});
    a.display();
    b.display();
    auto c = alg::mat_prod(a,b);
    c.display();
    c.apply([](alg::alg_type x) -> alg::alg_type {return x+2;});
    c.display();
    return 0;
}
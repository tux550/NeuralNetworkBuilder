#include <iostream>
#include <vector>
#include "algebra/alg.h"
#include "layers/fclayer.h"
using namespace std;

int main() {

    auto a = alg::Matrix( { {1,2,3}} );
    auto b = alg::Matrix( {{5,2},{4,4},{3,1}});
    cout << "INPUT" << endl;
    a.display();
    cout << "WEIGHTS" << endl;
    b.display();

    auto fc = ai::FCLayer(3.0,2.0);
    fc.set_input(a);
    fc.set_weights(b);
    fc.forward_propagation();
    auto c = fc.get_output();
    cout << "OUTPUT" << endl;
    c.display();

    /*
    auto c = alg::mat_prod(a,b);
    cout << "c1" << endl;
    c.display();
    c.apply([](alg::t_type x) -> alg::t_type {return x+2;});
    cout << "c2" << endl;
    c.display();
    */
    return 0;
}
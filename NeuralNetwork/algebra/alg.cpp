#include <iomanip>
#include "./alg.h"


namespace alg
{
    // Fill matrix
    void input_mat(std::istream& is, t_mat &m) {
        t_type tmp;
        for (auto r=0; r <arma::size(m)[0]; r++) {
            for (auto c=0; c <arma::size(m)[1]; c++) {
                is >> tmp;
                m(r,c) = tmp;
            }
        }
    }

    void output_mat(std::ostream& os, t_mat &m) {
        t_type tmp;
        for (auto r=0; r <arma::size(m)[0]; r++) {
            for (auto c=0; c <arma::size(m)[1]; c++) {
                os << std::setprecision(std::numeric_limits<double>::digits10 + 1) << m(r,c) << " ";
            }
            os << std::endl;
        }
    }
}

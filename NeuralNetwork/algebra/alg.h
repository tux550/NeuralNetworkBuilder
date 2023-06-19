#ifndef H_ALG
#define H_ALG

#include <iostream>
#include <armadillo>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <functional>

namespace alg
{
    // Matrix Types
    using t_dim  = unsigned long;
    using t_type = double;
    using t_mat  = arma::Mat<t_type>;


    // Function types
    using t_fmat  = std::function<void(t_type&)>;
    using t_m2t  = std::function<t_type(t_mat&)>;
    using t_m2m  = std::function<t_mat(t_mat&)>;
    using t_mm2t  = std::function<t_type(t_mat&,t_mat&)>;
    using t_mm2m  = std::function<t_mat(t_mat&,t_mat&)>;

    // Input types
    using vec_mat = std::vector<t_mat>;

    using t_inprow = std::vector<t_type>;
    using t_inpmat = std::vector<t_inprow>;

    // Fill matrix
    void input_mat(std::istream& is, t_mat &m);
}

#endif
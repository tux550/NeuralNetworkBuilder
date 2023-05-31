#ifndef H_ALG
#define H_ALG

#include <vector>
#include <stdexcept>
#include <iostream>
#include <functional>

namespace alg
{
    // Define types
    using alg_dim  = int;
    using alg_type = double;
    using alg_row  = std::vector<alg_type>;
    using alg_mat  = std::vector<alg_row>;

    // Define tensor class
    class Matrix
    {
        private:
            // Matrix
            alg_mat tensor;
        public:
            // CONSTRUCTOR
            Matrix(alg_dim _rows, alg_dim _cols);
            Matrix(alg_mat _tensor);
            // DESTRUCTOR
            ~Matrix();
            // Getters
            alg_dim get_rows();
            alg_dim get_cols();
            // Transpose
            Matrix transpose();
            // Apply function (element-wise)
            void apply(std::function<alg_type(alg_type)> func);
            // Display
            void display();

            // Friend functions
            friend Matrix mat_prod (Matrix,Matrix);
    };

    Matrix mat_prod(Matrix a, Matrix b);    
}

#endif
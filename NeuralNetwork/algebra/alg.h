#ifndef H_ALG
#define H_ALG

#include <vector>
#include <stdexcept>
#include <iostream>
#include <functional>

namespace alg
{
    // Define types
    using t_dim  = unsigned long;
    using t_type = double;
    using t_row  = std::vector<t_type>;
    using t_mat  = std::vector<t_row>;
    using t_t2t  = std::function<t_type(t_type)>;

    // Define tensor class
    class Matrix
    {
        private:
            // Matrix
            t_mat tensor;
        public:
            // CONSTRUCTOR
            Matrix(t_dim _rows, t_dim _cols);
            Matrix(t_mat _tensor);
            Matrix();
            // DESTRUCTOR
            ~Matrix();
            // Getters
            t_dim get_rows() const;
            t_dim get_cols() const;
            t_type get_val(t_dim r, t_dim c);
            // Setters
            void set_val(t_dim r, t_dim c, t_type val);
            // Transpose
            Matrix transpose();
            // Apply function (element-wise)
            Matrix apply(t_t2t func);
            // Sum
            t_type sum();
            // Display
            void display();


            Matrix operator*(const t_type&);
            Matrix operator+(const t_type&);
            Matrix operator*(const Matrix&);
            Matrix operator+(const Matrix&);
            Matrix operator-(const Matrix&);

            // Friend functions
            friend Matrix mat_prod (Matrix &a,Matrix &b);
    };

    Matrix mat_prod(Matrix &a, Matrix &b);    

    // Define extra types
    using t_mm2m  = std::function<Matrix(Matrix&,Matrix&)>;
    using vec_mat = std::vector<alg::Matrix>;
}

#endif
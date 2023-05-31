#include "alg.h"

namespace alg
{
    // Define types
    using alg_dim  = int;
    using alg_type = double;
    using alg_row  = std::vector<alg_type>;
    using alg_mat  = std::vector<alg_row>;

    Matrix::Matrix(alg_dim _rows, alg_dim _cols) {
        // Create tensor
        this->tensor = alg_mat(_rows, alg_row(_cols));
    }

    Matrix::Matrix(alg_mat _tensor) {
        // Create tensor
        this->tensor = _tensor;
    }
            
    // DESTRUCTOR
    Matrix::~Matrix() {
        // Delete tensor
    };

    // Getters
    alg_dim Matrix::get_rows() {
        if (tensor.size() == 0) return 0;
        return this->tensor.size();
    }

    alg_dim Matrix::get_cols() {
        if (tensor.size() == 0) return 0;
        return this->tensor[0].size();
    }

    // Transpose
    Matrix Matrix::transpose()
    {
        // Edge case: Empty tensor
        if (this->tensor.size() == 0) {
            throw std::invalid_argument("transpose: empty tensor");
        }

        // Temp vector
        auto cols = this->get_cols();
        auto rows = this->get_rows();
        Matrix transposed(cols, rows);

        // Populate vector
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                transposed.tensor[col][row] = this->tensor[row][col];
            }
        }

        // Return transposed
        return transposed;
    }

    // Apply function (element-wise)
    void Matrix::apply(std::function<alg_type(alg_type)> func) {
        for (auto &row : this->tensor) {
            for (auto &e : row) {
                e = func(e);
            }
        } 
    }

    // Display
    void Matrix::display() {
        for (auto &row : this->tensor) {
            for (auto &e : row) {
                std::cout << e << " ";
            }
            std::cout << std::endl;
        }
    }


    Matrix mat_prod(Matrix a, Matrix b)
    {
        // Validate shape
        if (a.get_cols() != b.get_rows()) {
            throw std::invalid_argument("mat_prod: invalid dimensions a(m x k1) @ b(k2 x n) -> k1!=k2.");
        }

        // Get dims
        alg_dim dim_m = a.get_rows();
        alg_dim dim_k = a.get_cols();
        alg_dim dim_n = b.get_cols();

        // Init result Tensor
        Matrix res(dim_m,dim_n);

        // Populate
        for (int i = 0; i < dim_m; i++) {
            for (int j = 0; j < dim_n; j++) {
                res.tensor[i][j] = 0;
                for (int k = 0; k < dim_k; k++) {
                    res.tensor[i][j] += a.tensor[i][k] * b.tensor[k][j];
                }
            }
        }

        // Return
        return res;
    }    
}

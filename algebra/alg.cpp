#include "alg.h"

namespace alg
{
    Matrix::Matrix(t_dim _rows, t_dim _cols) {
        // Create tensor
        this->tensor = t_mat(_rows, t_row(_cols));
    }

    Matrix::Matrix(t_mat _tensor) {
        // Create tensor
        this->tensor = _tensor;
    }
            
    // DESTRUCTOR
    Matrix::~Matrix() {
        // Delete tensor
    };

    // Getters
    t_dim Matrix::get_rows() const {
        if (tensor.size() == 0) return 0;
        return this->tensor.size();
    }

    t_dim Matrix::get_cols() const {
        if (tensor.size() == 0) return 0;
        return this->tensor[0].size();
    }

    t_type Matrix::get_val(t_dim r, t_dim c) {
        return tensor[r][c];
    }

    // Setters
    void Matrix::set_val(t_dim r, t_dim c, t_type val) {
        tensor[r][c] = val;
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
    Matrix Matrix::apply(t_t2t func) {
        // Temp vector
        auto cols = this->get_cols();
        auto rows = this->get_rows();
        Matrix transformed(rows, cols);

        // Populate vector
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                transformed.tensor[row][col] = func(tensor[row][col]);
            }
        }

        // Return transformed
        return transformed;
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

    Matrix Matrix::operator*(const t_type& x) {
        // Temp vector
        auto cols = this->get_cols();
        auto rows = this->get_rows();
        Matrix tmp(rows, cols);

        // Populate vector
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                tmp.tensor[row][col] = this->tensor[row][col] * x;
            }
        }

        // Return transposed
        return tmp;
    }

    Matrix Matrix::operator-(const Matrix& other) {
        // Temp vector
        if (get_rows() != other.get_rows()) {
            throw std::invalid_argument("matrix-: rows missmatch");
        }
        if (get_cols() != other.get_cols()) {
            throw std::invalid_argument("matrix-: cols missmatch");
        }
        auto cols = this->get_cols();
        auto rows = this->get_rows();
        Matrix tmp(rows, cols);

        // Populate vector
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                tmp.tensor[row][col] = this->tensor[row][col] - other.tensor[row][col];
            }
        }

        // Return transposed
        return tmp;
    }

    Matrix mat_prod(Matrix a, Matrix b)
    {
        // Validate shape
        if (a.get_cols() != b.get_rows()) {
            throw std::invalid_argument("mat_prod: invalid dimensions a(m x k1) @ b(k2 x n) -> k1!=k2.");
        }

        // Get dims
        t_dim dim_m = a.get_rows();
        t_dim dim_k = a.get_cols();
        t_dim dim_n = b.get_cols();

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

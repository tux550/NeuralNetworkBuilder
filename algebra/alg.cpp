#include <string>
#include "alg.h"

namespace alg
{
    MultidimMatrix::MultidimMatrix(t_dimvec _dimvec):
        tensor{},
        dimensions{ _dimvec }
        {
            // Validate dimensions
            if (_dimvec.size() == 0) {
                throw std::invalid_argument("MultidimMatrix: Cant create nondimensional Matrix");
            }
            // Get container size
            t_dim total_size = 1;
            for (auto &d : _dimvec) {
                if (d == 0) {
                    throw std::invalid_argument("MultidimMatrix: Dimension can not be 0");
                }
                total_size = total_size * d;
            }
            // Resize container
            tensor = t_container(total_size);
        }

    MultidimMatrix::MultidimMatrix(t_dimvec _dimvec, t_container &_tensor):
        tensor{ _tensor },
        dimensions{ _dimvec }
        {}

    MultidimMatrix MultidimMatrix::FromMat1D(t_mat1d &_mat_1d) {
        auto len = _mat_1d.size();
        t_dimvec dims = {len};
        MultidimMatrix res = MultidimMatrix(dims, _mat_1d );
        return res;
    } 
    
    MultidimMatrix MultidimMatrix::FromMat2D(t_mat2d &_mat_2d) {
        auto rows = _mat_2d.size();
        auto cols = _mat_2d[0].size();
        t_dimvec dims = {rows,cols};
        t_container tmp;
        for (auto &_mat_1d : _mat_2d) {
            for (auto &e : _mat_1d) {
                tmp.push_back(e);
            }
        }
        MultidimMatrix res = MultidimMatrix(dims, tmp);
        return res;
    }   

    MultidimMatrix MultidimMatrix::FromMat3D(t_mat3d &_mat_3d) {
        auto x_size = _mat_3d.size();
        auto y_size = _mat_3d[0].size();
        auto z_size = _mat_3d[0][0].size();
        t_dimvec dims = {x_size,y_size,z_size};
        t_container tmp;
        for (auto &_mat_2d : _mat_3d) {
            for (auto &_mat_1d : _mat_2d) {
                for (auto &e : _mat_1d) {
                    tmp.push_back(e);
                }
            }
        }
        MultidimMatrix res = MultidimMatrix(dims, tmp);
        return res;
    }   


    // DESTRUCTOR
    MultidimMatrix::~MultidimMatrix() = default;

    // Getters
    t_dim MultidimMatrix::get_ndims() const {
        return dimensions.size();
    }

    t_dimvec MultidimMatrix::get_shape() const {
        return dimensions;
    }

    t_dim MultidimMatrix::get_index(t_dimvec dims) {
        auto shape = get_shape();
        auto index = 0;
        auto factor = 1;
        for (long long d = shape.size()-1; d>= 0; d--) {
            index  += dims[d]*factor;
            factor *= (shape[d]);
        }
        return index;
    }

    t_dimvec MultidimMatrix::index_to_coords(t_dim index) {
        t_dimvec coords(get_ndims());
        auto shape = get_shape();
        auto factor = 1;
        for (long long d = shape.size() - 1; d>= 0; d--) {
            factor *= (shape[d]);
        }

        for (long long d = 0; d < shape.size(); d++) {
            factor = factor/shape[d];
            coords[d]  = index/factor;
            index = index%factor;

        }
        return coords;
    }


    t_type MultidimMatrix::get_val(t_dimvec dims) {
        return tensor[get_index(dims)];
    }

    // Setters
    void MultidimMatrix::set_val(t_dimvec dims, t_type val) {
        tensor[get_index(dims)] = val;
    }

    // Transpose
    MultidimMatrix MultidimMatrix::transpose(t_dim d1, t_dim d2)
    {
        // Tmp matrix
        t_dimvec shape = get_shape();
        t_dimvec new_shape = shape;
        new_shape[d1] = shape[d2];
        new_shape[d2] = shape[d1];

        MultidimMatrix res(new_shape);
        // Populate result
        t_dimvec coords(get_ndims());
        for (t_dim i = 0; i < tensor.size(); i++)
        {   // Get original coords
            auto coords = index_to_coords(i);
            // Swap
            auto tmp = coords[d1];
            coords[d1] = coords[d2];
            coords[d2] = tmp;
            // Save swapped
            res.set_val(coords,tensor[i]);
        }
        // Return result
        return res;
    }

    // Apply function (element-wise)
    MultidimMatrix MultidimMatrix::apply(t_t2t func) {
        // Tmp matrix
        t_dimvec shape = get_shape();
        MultidimMatrix res(shape);

        // Populate result
        t_dimvec coords(get_ndims());
        for (t_dim i = 0; i < tensor.size(); i++)
        {
            // Save swapped
            res.tensor[i] = func(tensor[i]);
        }

        // Return result
        return res;
    }
    // Sum
    t_type MultidimMatrix::sum() {
        t_type res = 0;
        for (t_dim i = 0; i < tensor.size(); i++)
        {
           res+=tensor[i];
        }
        return res;
    }
    // Display
    void MultidimMatrix::display() {
        for (t_dim i = 0; i < tensor.size(); i++) {
            std::cout << tensor[i] << " ";
        }
        std::cout << std::endl;
        /*
        // Counters
        t_dimvec shape = get_shape();
        std::vector<long long> counters(shape.size(), -1);

        for (t_dim i = 0; i < tensor.size(); i++)
        {
            // Init Brackets
            for (auto i=counters.size()-1; i>= 0; i--) {
                if (counters[i] == -1) {
                    std:: cout<< "[";
                    counters[i] = 0;
                }
                if (i==0) {break;}
            }
            // Print
            std::cout << tensor[i] << " ";
            // End Brackets
            counters[counters.size()-1]++;
            for (auto i=counters.size()-1; i>= 0; i--) {
                if (counters[i] == shape[i]) {
                    counters[i]=-1;
                    counters[i-1]++;
                    std:: cout<< "]";
                }
                if (i==0) {break;}
            }
        }
        */
    }

    MultidimMatrix MultidimMatrix::operator*(const t_type& x) {
        // Tmp matrix
        t_dimvec shape = get_shape();
        MultidimMatrix res(shape);

        // Populate vector
        for (t_dim i = 0; i < tensor.size(); i++)
        {
           res.tensor[i]=tensor[i]*x;
        }
        // Return result
        return res;
    }

    MultidimMatrix MultidimMatrix::operator+(const t_type& x) {
        // Tmp matrix
        t_dimvec shape = get_shape();
        MultidimMatrix res(shape);

        // Populate vector
        for (t_dim i = 0; i < tensor.size(); i++)
        {
           res.tensor[i]=tensor[i]+x;
        }
        // Return result
        return res;
    }

    MultidimMatrix MultidimMatrix::operator*(const MultidimMatrix& other) {
        // Validate shape
        if (get_shape() != other.get_shape()) {
            std::string error_message = "matrix*: Shape missmatch";
            throw std::invalid_argument(error_message);
        }
        // Tmp matrix
        t_dimvec shape = get_shape();
        MultidimMatrix res(shape);

        // Populate vector
        for (t_dim i = 0; i < tensor.size(); i++)
        {
           res.tensor[i]=tensor[i]*other.tensor[i];
        }
        // Return result
        return res;
    }

    MultidimMatrix MultidimMatrix::operator+(const MultidimMatrix& other) {
        // Validate shape
        if (get_shape() != other.get_shape()) {
            std::string error_message = "matrix+: Shape missmatch";
            throw std::invalid_argument(error_message);
        }
        // Tmp matrix
        t_dimvec shape = get_shape();
        MultidimMatrix res(shape);

        // Populate vector
        for (t_dim i = 0; i < tensor.size(); i++)
        {
           res.tensor[i]=tensor[i]+other.tensor[i];
        }
        // Return result
        return res;
    }

    MultidimMatrix MultidimMatrix::operator-(const MultidimMatrix& other) {
        // Validate shape
        if (get_shape() != other.get_shape()) {
            std::string error_message = "matrix-: Shape missmatch";
            throw std::invalid_argument(error_message);
        }
        // Tmp matrix
        t_dimvec shape = get_shape();
        MultidimMatrix res(shape);

        // Populate vector
        for (t_dim i = 0; i < tensor.size(); i++)
        {
           res.tensor[i]=tensor[i]-other.tensor[i];
        }
        // Return result
        return res;
    }

    MultidimMatrix mat_prod(MultidimMatrix &a, MultidimMatrix &b)
    {
        // TODO: https://www.iaeng.org/publication/WCE2010/WCE2010_pp1829-1833.pdf
    
        // Validate shape
        t_dimvec a_shape = a.get_shape();
        t_dimvec b_shape = b.get_shape();
        if (a_shape[1] != b_shape[0]) {
            throw std::invalid_argument("mat_prod: invalid dimensions");
        }

        // Get dims
        t_dim dim_m = a_shape[0];
        t_dim dim_k = a_shape[1];
        t_dim dim_n = b_shape[1];

        // Init result Tensor
        t_dimvec shape = {dim_m,dim_n};
        MultidimMatrix res(shape);

        // Populate
        for (t_dim i = 0; i < dim_m; i++) {
            for (t_dim j = 0; j < dim_n; j++) {
                auto tmp = 0;
                for (t_dim k = 0; k < dim_k; k++) {
                    tmp += a.get_val({i,k}) * b.get_val({k,j});
                }
                res.set_val({i,j},tmp);
            }
        }



        // Return
        return res;
    }

}


/*
int main() {
    std::cout << "TEST" << std::endl;
    alg::t_mat3d test {
        {
            { 1, 2, 3, 4 },
            { 5, 6, 2, 4},
            { 7, 8, 9, 3 }
        },
        {
            { 3, 2, 1, 4 },
            { 8, 8, 8, 8},
            { 9, 8, 9, 9 }
        },
    };

    alg::t_mat2d a = {
        {3,2},
        {2,5},
        {7,10}
    };
    alg::t_mat2d b = {
        {4,11},
        {6,9}
    };


    alg::MultidimMatrix A = alg::MultidimMatrix::FromMat2D(a);
    alg::MultidimMatrix B = alg::MultidimMatrix::FromMat2D(b);
    alg::MultidimMatrix TEST = alg::MultidimMatrix::FromMat3D(test);
    std::cout << "TEST INDEX TO COORDS" << std::endl;
    alg::t_dim x=13;
    auto t = TEST.index_to_coords(x) ;
    for (auto &e :t) {
        std::cout << e << " . "; 
    }
    std::cout << std::endl;



    std::cout << "MAT PRODUCT" << std::endl;
    auto C = alg::mat_prod(A,B);


    std::cout << "MAT TRANSPOSE" << std::endl;
    auto Ap = A.transpose();


    std::cout << "MAT DISPLAY" << std::endl;
    C.display();
    std::cout << "END" << std::endl;
    TEST.display();
    return 0;
}
*/


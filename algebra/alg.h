#ifndef H_ALG
#define H_ALG

#include <vector>
#include <stdexcept>
#include <iostream>
#include <functional>

namespace alg
{
    // Define types
    using t_dim       = unsigned long long;
    using t_type      = double;
    using t_dimvec    = std::vector<t_dim>;
    using t_container = std::vector<t_type>;
    using t_t2t       = std::function<t_type(t_type)>;

    using t_mat1d     = t_container;
    using t_mat2d     = std::vector<t_mat1d>;
    using t_mat3d     = std::vector<t_mat2d>;

    // Define tensor class
    class MultidimMatrix
    {
        private:
            // Matrix
            t_container tensor;
            t_dimvec    dimensions;
            t_dimvec index_to_coords(t_dim &index);
        public:
            t_dim  get_index(t_dimvec dims);
            // CONSTRUCTOR
            MultidimMatrix(t_dimvec _dimvec);
            MultidimMatrix(t_dimvec _dimvec, t_container &_tensor);
            static MultidimMatrix FromMat1D(t_mat1d &_mat_1d);
            static MultidimMatrix FromMat2D(t_mat2d &_mat_2d);
            static MultidimMatrix FromMat3D(t_mat3d &_mat_3d);
            // DESTRUCTOR
            ~MultidimMatrix();
            // Getters
            t_dim get_ndims() const;
            t_dimvec get_shape() const;
            t_type get_val(t_dimvec dims);
            // Setters
            void set_val(t_dimvec dims, t_type val);
            //void reshape(t_dimvec &dims);
            // Transpose
            MultidimMatrix transpose(t_dim d1=0, t_dim d2=1);
            // Apply function (element-wise)
            MultidimMatrix apply(t_t2t func);
            // Sum
            t_type sum();
            // Size
            t_dim size();
            // Display
            void display();


            MultidimMatrix operator*(const t_type&);
            MultidimMatrix operator+(const t_type&);
            MultidimMatrix operator*(const MultidimMatrix&);
            MultidimMatrix operator+(const MultidimMatrix&);
            MultidimMatrix operator-(const MultidimMatrix&);

            // Friend functions
            friend MultidimMatrix mat_prod (MultidimMatrix &a,MultidimMatrix &b, t_dim dim_a, t_dim dim_b);
    };

    MultidimMatrix mat_prod(MultidimMatrix &a, MultidimMatrix &b);    

    // Define extra types
    using t_mm2m  = std::function<MultidimMatrix(MultidimMatrix&,MultidimMatrix&)>;
}

#endif
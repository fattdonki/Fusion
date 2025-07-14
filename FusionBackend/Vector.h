#pragma once

#include <vector>
#include "Matrix.h"

struct Vector {
public:

    /*_______________________________Defenition_______________________________*/

    Vector(int size, const double* values = nullptr)
        : size_(size), data_(std::vector<double>(size, 0.0)) {
        if (values) {
            for (int i = 0; i < size; ++i) {
                data_[i] = values[i];
            }
        }
    }

    void set(int i, double value) {
        data_[i] = value;
    }

    double get(int i) {
        return data_[i];
    }

    /*_______________________________Operators_______________________________*/

    double& operator[](int index) { return data_[index]; }
    const double& operator[](int index) const { return data_[index]; }
    
    double operator*(const Vector& B) const {

        double sum = 0;
        for (int i = 0; i < size_; ++i) {
            sum += data_[i] * B[i];
        }
        return sum;
    }

    Vector operator*(const double s) const {

        Vector result = Vector::zero(size_);

        for (int i = 0; i < size_; ++i) {
            result[i] = data_[i] * s;
        }
        return result;
    }

    Vector operator/(const double s) const {

        Vector result = Vector::zero(size_);

        for (int i = 0; i < size_; ++i) {
            result[i] = data_[i] / s;
        }
        return result;
    }

    Vector operator+(const Vector& B) const {
        
        Vector result = Vector::zero(size_);

        for (int i = 0; i < size_; ++i) {
            result[i] += data_[i] + B[i];
        }
        return result;
    }

    Vector operator-(const Vector& B) const {

        Vector result = Vector::zero(size_);

        for (int i = 0; i < size_; ++i) {
            result[i] += data_[i] - B[i];
        }
        return result;
    }

    bool operator==(const Vector& B) const {
        if (size_ != B.size_) { return false; }

        for (int i = 0; i < size_; ++i) {
            if (data_[i] != B.data_[i]) { return false; }
        }
        return true;
    }

    bool operator!=(const Vector& B) const {
        return !(*this == B);
    }

    /*_______________________________Basic Functions_______________________________*/

    Vector normalize() const {
        return *this / magnitude();
    }

    double magnitude() const {
        double result = 0;
        for (int i = 0; i < size_; ++i) {
            result += std::pow(data_[i], 2);
        }
        return std::sqrt(result);
    }

private:
    int size_;
    std::vector<double> data_;

    static Vector zero(int n) {
        return Vector(n);
    }
};

Vector operator*(const Matrix& mat, const Vector& vec) {
    Vector result(mat.rows_, 0);

    for (int i = 0; i < mat.rows_; ++i) {
        for (int j = 0; j < mat.cols_; ++j) {
            result[i] += mat[i][j] * vec[j];
        }
    }
    return result;
}

Vector operator*(const Vector& vec, const Matrix& mat) {
    Vector result(mat.cols_, 0);

    for (int j = 0; j < mat.cols_; ++j) {
        for (int i = 0; i < mat.rows_; ++i) {
            result[j] += vec[i] * mat[i][j];
        }
    }
    return result;
}

extern "C" {

    /*_______________________________Defenition_______________________________*/

    __declspec(dllexport) void* Vector_create(int size, const double* values) {
        return new Vector(size, values);
    }

    __declspec(dllexport) void Vector_delete(void* vec) {
        delete static_cast<Vector*>(vec);
    }

    __declspec(dllexport) double Vector_find(void* vec, int index) {
        return static_cast<Vector*>(vec)->get(index);
    }

    /*_______________________________Operators_______________________________*/

    __declspec(dllexport) double Vector_mul(void* a, void* b){
        Vector* A = static_cast<Vector*>(a);
        Vector* B = static_cast<Vector*>(b);
        return *A * *B;
    }

    __declspec(dllexport) void* VecScalar_mul(void* a, double b) {
        Vector* A = static_cast<Vector*>(a);
        return new Vector(*A * b);
    }

    __declspec(dllexport) void* VecScalar_div(void* a, double b) {
        Vector* A = static_cast<Vector*>(a);
        return new Vector(*A / b);
    }

    __declspec(dllexport) void* VecMat_mul(void* vec, void* mat) {
        Vector* v = static_cast<Vector*>(vec);
        Matrix* M = static_cast<Matrix*>(mat);
        return new Vector(*v * *M);
    }

    __declspec(dllexport) void* MatVec_mul(void* vec, void* mat) {
        Vector* v = static_cast<Vector*>(vec);
        Matrix* M = static_cast<Matrix*>(mat);
        return new Vector(*M * *v);
    }

    __declspec(dllexport) void* Vector_add(void* a, void* b) {
        Vector* A = static_cast<Vector*>(a);
        Vector* B = static_cast<Vector*>(b);
        return new Vector(*A + *B);
    }

    __declspec(dllexport) void* Vector_sub(void* a, void* b) {
        Vector* A = static_cast<Vector*>(a);
        Vector* B = static_cast<Vector*>(b);
        return new Vector(*A - *B);
    }

    __declspec(dllexport) bool Vector_eqq(void* a, void* b) {
        Vector* A = static_cast<Vector*>(a);
        Vector* B = static_cast<Vector*>(b);
        return A == B;
    }

    __declspec(dllexport) bool Vector_neq(void* a, void* b) {
        Vector* A = static_cast<Vector*>(a);
        Vector* B = static_cast<Vector*>(b);
        return A != B;
    }

    /*_______________________________Basic Functions_______________________________*/

    __declspec(dllexport) void* Vector_normalize(void* a) {
        Vector* A = static_cast<Vector*>(a);
        return new Vector(A->normalize());
    }

    __declspec(dllexport) double Vector_magnitude(void* a) {
        Vector* A = static_cast<Vector*>(a);
        return A->magnitude();
    }

    /*_______________________________Static Functions_______________________________*/

    __declspec(dllexport) void* Vector_zero(int size) {
        return new Vector(size);
    }

}
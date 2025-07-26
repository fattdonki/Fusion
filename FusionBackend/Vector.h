#pragma once

#include <vector>
#include "Matrix.h"

extern PyTypeObject PyVectorType;

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

        Vector result(size_);

        for (int i = 0; i < size_; ++i) {
            result[i] = data_[i] * s;
        }
        return result;
    }

    Vector operator/(const double s) const {

        Vector result(size_);

        for (int i = 0; i < size_; ++i) {
            result[i] = data_[i] / s;
        }
        return result;
    }

    Vector operator+(const Vector& B) const {

        Vector result(size_);

        for (int i = 0; i < size_; ++i) {
            result[i] += data_[i] + B[i];
        }
        return result;
    }

    Vector operator-(const Vector& B) const {

        Vector result(size_);

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

    int size_;
    std::vector<double> data_;

    static Vector zero(int n) {
        return Vector(n);
    }

    std::string to_str() const {
        std::ostringstream oss;
        oss << std::defaultfloat;
        oss << "[";

        for (int i = 0; i < size_; ++i)
            oss << data_[i] << " ";

        oss << "]\n";
        return oss.str();
    }
};

Vector operator*(const Matrix& mat, const Vector& vec) {
    Vector result(mat.rows_, 0);

    for (int i = 0; i < mat.rows_; ++i) {
        for (int j = 0; j < mat.cols_; ++j) {
            result[i] += mat.data_[i * mat.cols_ + j] * vec[j];
        }
    }
    return result;
}

Vector operator*(const Vector& vec, const Matrix& mat) {
    Vector result(mat.cols_, 0);

    for (int j = 0; j < mat.cols_; ++j) {
        for (int i = 0; i < mat.rows_; ++i) {
            result[j] += vec[i] * mat.data_[i * mat.cols_ + j];
        }
    }
    return result;
}
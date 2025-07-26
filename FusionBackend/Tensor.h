#pragma once 

#include <vector>
#include <string>
#include <sstream>
#include <functional>

#undef _DEBUG
#include <Python.h>
#define _DEBUG

extern PyTypeObject PyTensorType;

struct Tensor {

    /*_______________________________Definition_______________________________*/

    Tensor(const std::vector<int>& shape, const double* values = nullptr)
        : shape_(shape), total_size_(1)
    {
        for (int dim : shape_) {
            total_size_ *= dim;
        }

        data_ = std::vector<double>(total_size_, 0.0);

        if (values) {
            for (int i = 0; i < total_size_; ++i) {
                data_[i] = values[i];
            }
        }
    }


    std::vector<int> shape_;
    int total_size_;
    std::vector<double> data_;

    const double& operator[](int i) const { return data_[i]; }
    double& operator[](int i) { return data_[i]; }

    static Tensor empty(std::vector<int> shape) { return Tensor(shape); }

    Tensor operator*(const Tensor& a) const {
        std::vector<int> new_shape = a.shape_;
        new_shape.insert(new_shape.end(), shape_.begin(), shape_.end());
        int new_total_size = a.total_size_ * total_size_;

        Tensor result(new_shape);

        for (int i = 0; i < a.total_size_; ++i) {
            for (int j = 0; j < total_size_; ++j) {
                result.data_[static_cast<std::vector<double, std::allocator<double>>::size_type>(i) * total_size_ + j] = a.data_[i] * data_[j];
            }
        }
        return result;
    }

    Tensor operator*(const double scalar) const {
        Tensor result(shape_);

        for (int i = 0; i < total_size_; ++i) 
            result.data_[static_cast<std::vector<double, std::allocator<double>>::size_type>(i)] = data_[i] * scalar;

        return result;
    }

    Tensor operator+(const Tensor& a) const {
        Tensor result(shape_);

        for (int i = 0; i < a.total_size_; ++i) 
            result.data_[i] = a.data_[i] + data_[i];

        return result;
    }

    Tensor operator-(const Tensor& a) const {
        Tensor result(shape_);

        for (int i = 0; i < a.total_size_; ++i) 
            result.data_[i] = a.data_[i] - data_[i];

        return result;
    }

    Tensor operator/(const double scalar) const {
        Tensor result(shape_);

        for (int i = 0; i < total_size_; ++i)
            result.data_[static_cast<std::vector<double, std::allocator<double>>::size_type>(i)] = data_[i] / scalar;

        return result;
    }

    bool operator==(const Tensor& a) const {
        return shape_ == a.shape_ && data_ == a.data_;
    }

    std::string to_str() const {
        std::ostringstream oss;
        oss << std::defaultfloat;

        std::function<void(int, int)> print_tensor = [&](int dim, int offset) {
            if (dim == shape_.size() - 1) {
                oss << "[ ";
                for (int i = 0; i < shape_[dim]; ++i) {
                    oss << data_[offset + i] << " ";
                }
                oss << "]";
            }
            else {
                oss << "[";
                int stride = 1;
                for (int k = dim + 1; k < shape_.size(); ++k) stride *= shape_[k];
                for (int i = 0; i < shape_[dim]; ++i) {
                    print_tensor(dim + 1, offset + i * stride);
                    if (i != shape_[dim] - 1) oss << ",\n";
                }
                oss << "]";
            }
        };

        if (shape_.empty()) {
            oss << "[ " << (data_.empty() ? 0.0 : data_[0]) << " ]";
        }
        else {
            print_tensor(0, 0);
        }
        oss << "\n";
        return oss.str();
    }
};
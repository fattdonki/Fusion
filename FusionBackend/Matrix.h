#pragma once

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#undef _DEBUG
#include <Python.h>
#define _DEBUG

std::vector<double> normalize(std::vector<double>& v) {
    std::vector<double> result(v.size(), 0);
    double magnitude = 0;
    for (int i = 0; i < v.size(); i++)
        magnitude += pow(v[i], 2);

    magnitude = sqrt(magnitude);

    for (int i = 0; i < v.size(); i++)
        result[i] = v[i] / magnitude;

    return result;
}

extern PyTypeObject PyMatrixType;

struct Matrix {

    /*_______________________________Definition_______________________________*/

    Matrix(int rows, int cols, const double* values = nullptr)
        : rows_(rows), cols_(cols)
    {
        if (values)
            data_.resize(rows * cols);
        else
            data_.assign(rows * cols, 0.0);

        if (values)
            std::copy(values, values + rows * cols, data_.begin());
    }

    Matrix(int rows, int cols, const std::vector<double>& values)
        : rows_(rows), cols_(cols), data_(values) {
        if (values.size() != static_cast<size_t>(rows * cols)) {
            throw std::invalid_argument("Data size does not match matrix dimensions");
        }
    }


    void set(int i, int j, double value) { data_[i * cols_ + j] = value; }
    double get(int i, int j) const { return data_[i * cols_ + j]; }

    std::vector<double> data_;
    int rows_, cols_;

    static Matrix zero(int n, int h) { return Matrix(n, h); }

    static Matrix identity(int n) {
        Matrix result(n, n);
        for (int i = 0; i < n; ++i) result.data_[i * result.cols_ + i] = 1;
        return result;
    }

    static Matrix Rx(int angle) {
        Matrix m(3,3, {1, 0, 0,
            0, cos(angle), -sin(angle),
            0, sin(angle), cos(angle)});
            return Matrix(m);
    }

    static Matrix Ry(int angle) {
        Matrix m(3, 3, { cos(angle), 0, sin(angle),
            0, 1, 0,
            -sin(angle), 0, cos(angle) });
        return Matrix(m);
    }

    static Matrix Rz(int angle) {
        Matrix m(3, 3, { cos(angle), -sin(angle), 0,
            sin(angle), cos(angle), 0,
            0, 0, 1 });
        return Matrix(m);
    }

    /*_______________________________Operators_______________________________*/

    Matrix operator*(const Matrix& B) const {
        Matrix result(rows_, B.cols_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < B.cols_; ++j) {
                for (int k = 0; k < cols_; ++k) {
                    result.data_[i * result.cols_ + j] += data_[i * cols_ + k] * B.data_[k * B.cols_ + j];
                }
            }
        }
        return result;
    }

    Matrix operator*(const double s) const {
        Matrix result(rows_, cols_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                result.data_[i * result.cols_ + j] = data_[i * cols_ + j] * s;
            }
        }
        return result;
    }

    Matrix operator+(const Matrix& B) const {
        Matrix result(rows_, cols_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                result.data_[i * result.cols_ + j] = data_[i * cols_ + j] + B.data_[i * B.cols_ + j];
            }
        }
        return result;
    }

    Matrix operator-(const Matrix& B) const {
        Matrix result(rows_, cols_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                result.data_[i * cols_ + j] = data_[i * cols_ + j] - B.data_[i * B.cols_ + j];
            }
        }
        return result;
    }

    bool operator==(const Matrix& B) const {
        if (rows_ != B.rows_ || cols_ != B.cols_) return false;
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                if (data_[i * cols_ + j] != B.data_[i * B.cols_ + j]) return false;
            }
        }
        return true;
    }

    bool operator!=(const Matrix& B) const { return !(*this == B); }

    /*_______________________________Functions_______________________________*/

    double trace() const {
        double result = 0.0;
        for (int i = 0; i < rows_; ++i) result += data_[i * cols_ + i];
        return result;
    }

    Matrix transpose() const {
        Matrix result(cols_, rows_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                result.data_[j * result.cols_ + i] = data_[i * cols_ + j];
            }
        }
        return result;
    }

    std::tuple<Matrix, Matrix> LUD() const {

        int rows = rows_, cols = cols_;

        Matrix L(rows, rows);
        Matrix U(rows, rows);

        for (int i = 0; i < rows; i++) {

            for (int j = i; j < rows; j++) {
                double sum = 0;
                for (int k = 0; k < i; k++) {
                    sum += L.data_[i * L.cols_ + k] * U.data_[k * U.cols_ + j];
                }
                U.data_[i * U.cols_ + j] = data_[i * cols_ + j] - sum;
            }

            for (int j = i; j < rows; j++) {
                if (i == j) {
                    L.data_[i * cols_ + i] = 1;
                }
                else {
                    double sum = 0;
                    for (int k = 0; k < i; k++)
                        sum += L.data_[j * L.cols_ + k] * U.data_[k * U.cols_ + j];
                    L.data_[j * cols_ + i] = (data_[j * cols_ + i] - sum) / U.data_[i * U.cols_ + i];
                }
            }
        }
        return { L, U };
    }

    std::tuple<Matrix, Matrix> QRD() const {
        Matrix Q(rows_, cols_);
        Matrix R(cols_, cols_);
        for (int j = 0; j < cols_; ++j) {
            std::vector<double> v(rows_, 0);
            for (int i = 0; i < rows_; ++i) v[i] = data_[i * cols_ + j];
            for (int i = 0; i < j; ++i) {
                R.data_[i * R.cols_ + j] = 0;
                for (int k = 0; k < rows_; ++k) R.data_[i * R.cols_ + j] += Q.data_[k * Q.cols_ + i] * data_[k * cols_ + j];
                for (int k = 0; k < rows_; ++k) v[k] -= R.data_[i * R.cols_ + j] * Q.data_[k * Q.cols_ + i];
            }
            R.data_[j * R.cols_ + j] = 0;
            for (int k = 0; k < rows_; ++k) R.data_[j * R.cols_ + j] += v[k] * v[k];
            R.data_[j * cols_ + j] = std::sqrt(R.data_[j * R.cols_ + j]);
            if (R.data_[j * R.cols_ + j] > 0) {
                for (int k = 0; k < rows_; ++k) Q.data_[k * Q.cols_ + j] = v[k] / R.data_[j * R.cols_ + j];
            }
        }
        return { Q, R };
    }

std::tuple<Matrix, Matrix, Matrix> SVD() const {
        Matrix A = *this;
        Matrix At = A.transpose();
        Matrix AtA = At * A;
        Matrix AAt = A * At;

        std::vector<double> ataEigenVals = AtA.eigenvalues();
        std::vector<std::vector<double>> ataEigenVecs = AtA.eigenvectors();
        std::vector<std::vector<double>> aatEigenVecs = AAt.eigenvectors();

        std::vector<int> sorted_indices(ataEigenVals.size());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(),
            [&](int i1, int i2) { return ataEigenVals[i1] > ataEigenVals[i2]; });

        Matrix sigma(A.rows_, A.cols_);
        for (int i = 0; i < sorted_indices.size(); ++i) {
            double value = std::max(0.0, static_cast<double>(ataEigenVals[sorted_indices[sorted_indices.size() - (static_cast<unsigned long long>(i) + 1)]]));
            sigma.data_[i * sigma.cols_ + i] = std::sqrt(value);
        }

        Matrix U(A.rows_, A.rows_);
        Matrix V(A.cols_, A.cols_);
        for (int i = 0; i < std::min(A.rows_, A.cols_); ++i) {
            if (i < aatEigenVecs.size()) {
                std::vector<double> u_vec = normalize(aatEigenVecs[sorted_indices[i]]);
                for (int j = 0; j < A.rows_; ++j) 
                    U.data_[j * U.cols_ + i] = u_vec[j];
            }
            if (i < ataEigenVecs.size()) {
                std::vector<double> v_vec = normalize(ataEigenVecs[sorted_indices[i]]);
                if (i == 0) {
                    for (int j = 0; j < A.cols_; ++j)
                        V.data_[j * V.cols_ + 0] = v_vec[j] * -1;
                }
                else {
                    for (int j = 0; j < A.cols_; ++j)
                        V.data_[j * V.cols_ + i] = v_vec[j];
                }
            }
        }

        return { U, sigma, V };
    }

    std::vector<double> eigenvalues() const {
        Matrix H = *this;
        std::vector<double> values(rows_, 0);
        for (int i = 0; i < 1000; i++) {
            auto [Q,R] = H.QRD();
            H = R * Q;
            double offDiagSum = 0;
            for (int j = 0; j < rows_; ++j) 
                for (int k = 0; k < rows_; ++k) 
                    if (j != k) offDiagSum += std::abs(H.data_[j * H.cols_ + k]);

            if (offDiagSum < 1e-10) break;
        }
        for (int i = 0; i < rows_; ++i) values[i] = H.data_[i * H.cols_ + i];
        return values;
    }

    Matrix RREF() const {
        Matrix result = *this;
        int lead = 0;
        int rows = result.rows_;
        int cols = result.cols_;
        const double epsilon = 1e-6;
        for (int r = 0; r < rows && lead < cols; r++) {
            int i = r;
            while (std::abs(result.data_[i * result.cols_ + lead]) < epsilon) {
                i++;
                if (i == rows) {
                    i = r;
                    lead++;
                    if (lead == cols) return result;
                }
            }
            if (i != r) for (int j = 0; j < result.cols_; ++j)
                std::swap(result.data_[i * result.cols_ + j], result.data_[r * result.cols_ + j]);
            if (std::abs(result.data_[r * result.cols_ + lead]) > epsilon) {
                double pivot = result.data_[r * result.cols_ + lead];
                for (int j = 0; j < cols; j++) result.data_[r * result.cols_ + j] /= pivot;
                for (int k = 0; k < rows; k++) {
                    if (k != r && std::abs(result.data_[k * cols_ + lead]) > epsilon) {
                        double factor = result.data_[k * cols_ + lead];
                        for (int j = 0; j < cols; j++) result.data_[k * cols_ + j] -= factor * result.data_[r * cols_ + j];
                    }
                }
            }
            lead++;
        }
        return result;
    }

    std::vector<std::vector<double>> null_space() const {
        Matrix rref = RREF();
        std::vector<int> free_cols;
        std::vector<int> pivot_cols;
        const double epsilon = 1e-6;
        int r = 0;
        for (int c = 0; c < cols_ && r < rows_; c++) {
            bool found_pivot = false;
            for (int i = r; i < rows_; i++) {
                if (std::abs(rref.data_[i * cols_ + c]) > epsilon) {
                    if (i != r) for (int j = 0; j < rref.cols_; ++j)
                        std::swap(rref.data_[i * rref.cols_ + j], rref.data_[r * rref.cols_ + j]);
                    pivot_cols.push_back(c);
                    r++;
                    found_pivot = true;
                    break;
                }
            }
            if (!found_pivot) free_cols.push_back(c);
        }
        for (int c = r; c < cols_; c++) free_cols.push_back(c);
        std::vector<std::vector<double>> basis;
        for (int j : free_cols) {
            std::vector<double> v(cols_, 0.0);
            v[j] = 1.0;
            for (int i = 0; i < r; i++) {
                for (int c = 0; c < cols_; c++) {
                    if (std::abs(rref.data_[i * cols_ + c]) > epsilon) {
                        v[c] = -rref.data_[i * cols_ + j];
                        break;
                    }
                }
            }
            basis.push_back(v);
        }
        return basis;
    }

    std::vector<std::vector<double>> eigenvectors() const {
        std::vector<double> eigenvals = this->eigenvalues();
        std::vector<std::vector<double>> result;
        struct VectorCompare {
            bool operator()(const std::vector<double>& a, const std::vector<double>& b) const {
                const double epsilon = 1e-6;
                for (size_t i = 0; i < a.size(); i++) {
                    if (std::abs(a[i] - b[i]) > epsilon) return a[i] < b[i];
                }
                return false;
            }
        };
        std::set<std::vector<double>, VectorCompare> unique_vectors;
        for (double lambda : eigenvals) {
            Matrix shifted = *this - (Matrix::identity(this->rows_) * lambda);
            std::vector<std::vector<double>> basis = shifted.null_space();
            for (auto& vec : basis) {
                unique_vectors.insert(vec);
            }
        }
        result.assign(unique_vectors.begin(), unique_vectors.end());
        return result;
    }

    double determinant() const {
        int n = rows_;

        if (n == 2) {
            return data_[0] * data_[cols_ + 1] - data_[1] * data_[cols_];
        }

        double result = 0;
        for (int j = 0; j < n; ++j) {
            int coef = (j % 2 == 0) ? 1 : -1;

            Matrix submatrix(n - 1, n - 1);
            for (int i = 1; i < n; ++i) {
                int sub_col = 0;
                for (int k = 0; k < n; ++k) {
                    if (k == j) continue;
                    submatrix.data_[(i - 1) * cols_ + (sub_col)] = data_[i * cols_ + k];
                    ++sub_col;
                }
            }
            result += coef * data_[j] * submatrix.determinant();
        }

        return result;
    }

    std::string to_str() const {
        std::ostringstream oss;
        oss << std::defaultfloat;
        oss << "[";

        for (int i = 0; i < rows_; ++i) {
            oss << "[ ";
            for (int j = 0; j < cols_; ++j) 
                oss << data_[i * cols_ + j] << " ";
            
            oss << "]\n";
        }

        oss.seekp(-1, std::ios_base::end);
        oss << "]";
        oss << "\n";
        return oss.str();
    }
};
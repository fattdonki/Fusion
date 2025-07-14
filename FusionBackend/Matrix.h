#pragma once

#include <vector>
#include <stdexcept>
#include <string>
#include <cmath>
#include <algorithm>
#include <set>
#include <numeric>

struct Pair {
    void* A;
    void* B;
    void* returnA() { return A; }
    void* returnB() { return B; }
};

struct Triple {
    void* A;
    void* B;
    void* C;
    void* returnA() { return A; }
    void* returnB() { return B; }
    void* returnC() { return C; }
};

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

struct Matrix {
public:

    /*_______________________________Defenition_______________________________*/

    Matrix(int rows, int cols, const double* values = nullptr)
        : rows_(rows), cols_(cols), data_(rows, std::vector<double>(cols, 0.0)) {
        if (values) {
            for (int i = 0; i < rows_; ++i)
                for (int j = 0; j < cols_; ++j)
                    data_[i][j] = values[i * cols_ + j];
        }
    }

    void set(int i, int j, double value) { data_[i][j] = value; }
    double get(int i, int j) { return data_[i][j]; }

    const std::vector<double>& operator[](int i) const { return data_[i]; }
    std::vector<double>& operator[](int i) { return data_[i]; }

    /*_______________________________Operators_______________________________*/

    Matrix operator*(const Matrix& B) const {
        Matrix result = Matrix::zero(rows_, B.cols_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < B.cols_; ++j) {
                for (int k = 0; k < cols_; ++k) {
                    result[i][j] += data_[i][k] * B.data_[k][j];
                }
            }
        }
        return result;
    }

    Matrix operator*(const double s) const {
        Matrix result = Matrix::zero(rows_, cols_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                result[i][j] = data_[i][j] * s;
            }
        }
        return result;
    }

    Matrix operator+(const Matrix& B) const {
        Matrix result = Matrix::zero(rows_, cols_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                result[i][j] = data_[i][j] + B[i][j];
            }
        }
        return result;
    }

    Matrix operator-(const Matrix& B) const {
        Matrix result = Matrix::zero(rows_, cols_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                result[i][j] = data_[i][j] - B[i][j];
            }
        }
        return result;
    }

    bool operator==(const Matrix& B) const {
        if (rows_ != B.rows_ || cols_ != B.cols_) return false;
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                if (data_[i][j] != B[i][j]) return false;
            }
        }
        return true;
    }

    bool operator!=(const Matrix& B) const { return !(*this == B); }

    /*_______________________________Functions_______________________________*/

    double trace() const {
        double result = 0.0;
        for (int i = 0; i < rows_; ++i) result += data_[i][i];
        return result;
    }

    Matrix transpose() const {
        Matrix result = Matrix::zero(cols_, rows_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                result[j][i] = data_[i][j];
            }
        }
        return Matrix(result);
    }

    Pair LUD() const {

        int rows = rows_, cols = cols_;

        Matrix L = zero(rows, rows);
        Matrix U = zero(rows, rows);

        for (int i = 0; i < rows; i++) {

            for (int j = i; j < rows; j++) {
                double sum = 0;
                for (int k = 0; k < i; k++) {
                    sum += L[i][k] * U[k][j];
                }
                U[i][j] = data_[i][j] - sum;
            }

            for (int j = i; j < rows; j++) {
                if (i == j) {
                    L[i][i] = 1;
                }
                else {
                    double sum = 0;
                    for (int k = 0; k < i; k++) {
                        sum += L[j][k] * U[k][j];
                    }
                    L[j][i] = (data_[j][i] - sum) / U[i][i];
                }
            }
        }
        return { new Matrix(L), new Matrix(U) };
    }

    Pair QRD() const {
        Matrix Q = Matrix::zero(rows_, cols_);
        Matrix R = Matrix::zero(cols_, cols_);
        for (int j = 0; j < cols_; ++j) {
            std::vector<double> v(rows_, 0);
            for (int i = 0; i < rows_; ++i) v[i] = data_[i][j];
            for (int i = 0; i < j; ++i) {
                R[i][j] = 0;
                for (int k = 0; k < rows_; ++k) R[i][j] += Q[k][i] * data_[k][j];
                for (int k = 0; k < rows_; ++k) v[k] -= R[i][j] * Q[k][i];
            }
            R[j][j] = 0;
            for (int k = 0; k < rows_; ++k) R[j][j] += v[k] * v[k];
            R[j][j] = std::sqrt(R[j][j]);
            if (R[j][j] > 0) {
                for (int k = 0; k < rows_; ++k) Q[k][j] = v[k] / R[j][j];
            }
        }
        return { new Matrix(Q), new Matrix(R) };
    }

    Triple SVD() const {
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

        Matrix sigma = Matrix::zero(A.rows_, A.cols_);
        for (int i = 0; i < sorted_indices.size(); ++i) {
            double value = std::max(0.0, static_cast<double>(ataEigenVals[sorted_indices[sorted_indices.size() - (i + 1)]]));
            sigma[i][i] = std::sqrt(value);
        }

        Matrix U = Matrix::zero(A.rows_, A.rows_);
        Matrix V = Matrix::zero(A.cols_, A.cols_);
        for (int i = 0; i < std::min(A.rows_, A.cols_); ++i) {
            if (i < aatEigenVecs.size()) {
                std::vector<double> u_vec = normalize(aatEigenVecs[sorted_indices[i]]);
                for (int j = 0; j < A.rows_; ++j) {
                    U[j][i] = u_vec[j];
                }
            }
            if (i < ataEigenVecs.size()) {
                std::vector<double> v_vec = normalize(ataEigenVecs[sorted_indices[i]]);
                if (i == 0) {
                    for (int j = 0; j < A.cols_; ++j)
                        V[j][0] = v_vec[j] * -1;
                }
                else {
                    for (int j = 0; j < A.cols_; ++j)
                        V[j][i] = v_vec[j];
                }
            }
        }

        return { new Matrix(U), new Matrix(sigma), new Matrix(V) };
    }

    std::vector<double> eigenvalues() const {
        Matrix H = *this;
        std::vector<double> values(rows_, 0);
        for (int i = 0; i < 1000; i++) {
            Pair QR = H.QRD();
            Matrix* Q = static_cast<Matrix*>(QR.returnA());
            Matrix* R = static_cast<Matrix*>(QR.returnB());
            H = *R * *Q;
            double offDiagSum = 0;
            for (int j = 0; j < rows_; ++j) {
                for (int k = 0; k < rows_; ++k) {
                    if (j != k) offDiagSum += std::abs(H[j][k]);
                }
            }
            if (offDiagSum < 1e-10) break;
        }
        for (int i = 0; i < rows_; ++i) values[i] = H[i][i];
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
            while (std::abs(result[i][lead]) < epsilon) {
                i++;
                if (i == rows) {
                    i = r;
                    lead++;
                    if (lead == cols) return result;
                }
            }
            if (i != r) std::swap(result[i], result[r]);
            if (std::abs(result[r][lead]) > epsilon) {
                double pivot = result[r][lead];
                for (int j = 0; j < cols; j++) result[r][j] /= pivot;
                for (int k = 0; k < rows; k++) {
                    if (k != r && std::abs(result[k][lead]) > epsilon) {
                        double factor = result[k][lead];
                        for (int j = 0; j < cols; j++) result[k][j] -= factor * result[r][j];
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
                if (std::abs(rref[i][c]) > epsilon) {
                    if (i != r) std::swap(rref[i], rref[r]);
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
                    if (std::abs(rref[i][c]) > epsilon) {
                        v[c] = -rref[i][j];
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
            return data_[0][0] * data_[1][1] - data_[0][1] * data_[1][0];
        }

        double result = 0;
        for (int j = 0; j < n; ++j) {
            int coef = (j % 2 == 0) ? 1 : -1;

            Matrix submatrix = Matrix::zero(n - 1, n - 1);
            for (int i = 1; i < n; ++i) {
                int sub_col = 0;
                for (int k = 0; k < n; ++k) {
                    if (k == j) continue;
                    submatrix[i - 1][sub_col] = data_[i][k];
                    ++sub_col;
                }
            }
            result += coef * data_[0][j] * submatrix.determinant();
        }

        return result;
    }

    int rows_, cols_;
    std::vector<std::vector<double>> data_;
    static Matrix zero(int n, int h) { return Matrix(n, h); }
    static Matrix identity(int n) {
        Matrix result = Matrix::zero(n, n);
        for (int i = 0; i < n; ++i) result[i][i] = 1;
        return result;
    }
};

extern "C" {

    /*_______________________________Defenition_______________________________*/

    __declspec(dllexport) void* Matrix_create(int rows, int cols, const double* values) {
        return new Matrix(rows, cols, values);
    }
    __declspec(dllexport) void Matrix_delete(void* mat) {
        delete static_cast<Matrix*>(mat);
    }
    __declspec(dllexport) double Matrix_find(void* mat, int i, int j) {
        return static_cast<Matrix*>(mat)->get(i, j);
    }

    /*_______________________________Operators_______________________________*/

    __declspec(dllexport) void* Matrix_mul(void* a, void* b) {
        Matrix* A = static_cast<Matrix*>(a);
        Matrix* B = static_cast<Matrix*>(b);
        return new Matrix(*A * *B);
    }
    __declspec(dllexport) void* MatScalar_mul(void* a, double b) {
        Matrix* A = static_cast<Matrix*>(a);
        return new Matrix(*A * b);
    }
    __declspec(dllexport) void* Matrix_add(void* a, void* b) {
        Matrix* A = static_cast<Matrix*>(a);
        Matrix* B = static_cast<Matrix*>(b);
        return new Matrix(*A + *B);
    }
    __declspec(dllexport) void* Matrix_sub(void* a, void* b) {
        Matrix* A = static_cast<Matrix*>(a);
        Matrix* B = static_cast<Matrix*>(b);
        return new Matrix(*A - *B);
    }
    __declspec(dllexport) bool Matrix_eqq(void* a, void* b) {
        Matrix* A = static_cast<Matrix*>(a);
        Matrix* B = static_cast<Matrix*>(b);
        return *A == *B;
    }
    __declspec(dllexport) bool Matrix_neq(void* a, void* b) {
        Matrix* A = static_cast<Matrix*>(a);
        Matrix* B = static_cast<Matrix*>(b);
        return *A != *B;
    }

    /*_______________________________Basic Functions_______________________________*/

    __declspec(dllexport) double Matrix_trace(void* mat) {
        return static_cast<Matrix*>(mat)->trace();
    }
    __declspec(dllexport) void* Matrix_transpose(void* mat) {
        Matrix* M = static_cast<Matrix*>(mat);
        return new Matrix(M->transpose());
    }

    __declspec(dllexport) void* Matrix_LUD(void* mat) {
        Matrix* M = static_cast<Matrix*>(mat);
        return new Pair(M->LUD());
    }

    __declspec(dllexport) void* Matrix_QRD(void* mat) {
        Matrix* M = static_cast<Matrix*>(mat);
        return new Pair(M->QRD());
    }

    __declspec(dllexport) void* Matrix_SVD(void* mat) {
        Matrix* M = static_cast<Matrix*>(mat);
        return new Triple(M->SVD());
    }

    __declspec(dllexport) void* Matrix_Eigenvalues(void* mat) {
        Matrix* M = static_cast<Matrix*>(mat);
        return new std::vector<double>(M->eigenvalues());
    }
    __declspec(dllexport) void* Matrix_Eigenvectors(void* mat) {
        Matrix* M = static_cast<Matrix*>(mat);
        return new std::vector<std::vector<double>>(M->eigenvectors());
    }
    __declspec(dllexport) void* Matrix_null_space(void* mat) {
        Matrix* M = static_cast<Matrix*>(mat);
        return new std::vector<std::vector<double>>(M->null_space());
    }
    __declspec(dllexport) void* Matrix_RREF(void* mat) {
        Matrix* M = static_cast<Matrix*>(mat);
        return new Matrix(M->RREF());
    }

    __declspec(dllexport) double Matrix_determinant(void* mat) {
        Matrix* M = static_cast<Matrix*>(mat);
        return M->determinant();
    }

    /*_______________________________Static Functions_______________________________*/

    __declspec(dllexport) void* Matrix_identity(int size) {
        Matrix* m = new Matrix(size, size);
        for (int i = 0; i < size; ++i) m->set(i, i, 1.0);
        return m;
    }
    __declspec(dllexport) void* Matrix_zero(int rows, int cols) {
        return new Matrix(rows, cols);
    }

    /*_______________________________Pairs and Triples_______________________________*/

    __declspec(dllexport) void* MatrixPair_A(void* pair) {
        Pair* P = static_cast<Pair*>(pair);
        return P->returnA();
    }
    __declspec(dllexport) void* MatrixPair_B(void* pair) {
        Pair* P = static_cast<Pair*>(pair);
        return P->returnB();
    }

    __declspec(dllexport) void* MatrixTriple_A(void* triple) {
        Triple* T = static_cast<Triple*>(triple);
        return T->returnA();
    }

    __declspec(dllexport) void* MatrixTriple_B(void* triple) {
        Triple* T = static_cast<Triple*>(triple);
        return T->returnB();
    }

    __declspec(dllexport) void* MatrixTriple_C(void* triple) {
        Triple* T = static_cast<Triple*>(triple);
        return T->returnC();
    }
}
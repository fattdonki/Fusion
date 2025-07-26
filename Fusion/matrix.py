from .backend_bindings import lib
import ctypes
import math
import random
import array

class Matrix:
    def __init__(self, data):
        if __debug__:
            if not all(isinstance(row, list) for row in data):
                raise ValueError("Data must be a non-empty list of lists.")
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0])
        flat = array.array('d', (v for row in data for v in row))
        arr = (ctypes.c_double * len(flat)).from_buffer(flat)
        self.obj = lib.Matrix_create(self.rows, self.cols, arr)
        
    def __del__(self):
        if hasattr(self, "obj"):
            lib.Matrix_delete(self.obj)

    def __getitem__(self, index):
        if isinstance(index, int):
            if (index < 0 or index >= self.rows):
                raise IndexError("Row index out of bounds.")

            return self.data[index]
        
        elif isinstance(index, tuple):
            if len(index) != 2:
                raise IndexError("Index must be a tuple of (row, column).")
            row, col = index
            if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
                raise IndexError("Row or column index out of bounds.")
            return self.data[row][col]

    def __str__(self):
        self.data = self.__fill_data()
        result = "[\n"
        for row in range(self.rows):
            for value in range(self.cols):
                result += str(self.data[row][value]) + ", "
            result = result[:-2]  
            result += "\n"
        result += "]"
        return result
    
    def __repr__(self):
        return self.__str__()
    
    def __name__(self):
        return "Matrix"

    def __mul__(self, other):
        if isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError("Matrix dimensions do not match for multiplication.")
            
            result_obj = lib.Matrix_mul(self.obj, other.obj)
            result = Matrix.__new__(Matrix)
            result.obj = result_obj
            result.rows = self.rows
            result.cols = other.cols
            return result
        
        elif isinstance(other, (int, float)):
            result_obj = lib.MatScalar_mul(self.obj, ctypes.c_double(other))
            result = Matrix.__new__(Matrix)
            result.obj = result_obj
            result.rows = self.rows
            result.cols = self.cols
            return result
        
        
        return NotImplemented
        
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            result_obj = lib.MatScalar_mul(self.obj, ctypes.c_double(other))
            result = Matrix.__new__(Matrix)
            result.obj = result_obj
            result.rows = self.rows
            result.cols = self.cols
            return result

        return NotImplemented    
    
    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions for addition.")
        
        result_obj = lib.Matrix_add(self.obj, other.obj)
        result = Matrix.__new__(Matrix)
        result.obj = result_obj
        result.rows = self.rows
        result.cols = self.cols
        return result
    
    def __sub__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions for subtraction.")
        
        result_obj = lib.Matrix_sub(self.obj, other.obj)
        result = Matrix.__new__(Matrix)
        result.obj = result_obj
        result.rows = self.rows
        result.cols = self.cols
        return result
    
    def __eq__(self, other):
        return lib.Matrix_eqq(self.obj, other.obj)
    
    def __ne__(self, other):
        return lib.Matrix_neq(self.obj, other.obj)
    
    def __fill_data(self):
        return [[round(lib.Matrix_find(self.obj, j, i),3) for i in range(self.cols)] for j in range(self.rows)]
    
    @staticmethod
    def identity(n):
        obj = lib.Matrix_identity(n)
        m = Matrix.__new__(Matrix)
        m.obj = obj
        m.rows = n
        m.cols = n
        m.data = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        return m

    @staticmethod
    def zero(rows, cols):
        obj = lib.Matrix_zero(rows, cols)
        m = Matrix.__new__(Matrix)
        m.obj = obj
        m.rows = rows
        m.cols = cols
        m.data = [[0 for _ in range(cols)] for _ in range(rows)]
        return m
    
    @staticmethod
    def random(rows, cols, min_val=0, max_val=1):
        if min_val >= max_val:
            raise ValueError("min_val must be less than max_val.")

        data = [[round(random.uniform(min_val, max_val), 2) for _ in range(cols)] for _ in range(rows)]
        return Matrix(data)
    
    @staticmethod
    def Rx(angle):
        data = [[1, 0, 0], 
            [0, round(math.cos(angle),3), round(-math.sin(angle),3)],
            [0, round(math.sin(angle),3), round(math.cos(angle),3)]]
        return Matrix(data)
    
    @staticmethod
    def Ry(angle): 
        data = [[round(math.cos(angle),3), 0, round(math.sin(angle),3)], 
                [0, 1, 0], 
                [round(-math.sin(angle),3), 0, round(math.cos(angle),3)]]
        return Matrix(data)
    
    @staticmethod
    def Rz(angle):
        data = [[round(math.cos(angle),3), round(-math.sin(angle),3), 0], 
                [round(math.sin(angle),3), round(math.cos(angle),3), 0], 
                [0, 0, 1]]
        return Matrix(data)
    
    def trace(self):
        if self.rows != self.cols:
            raise ValueError("Trace can only be computed for square matrices.")
        return lib.Matrix_trace(self.obj)
    
    def transpose(self):
        obj = lib.Matrix_transpose(self.obj)
        m = Matrix.__new__(Matrix)
        m.obj = obj
        m.rows = self.cols
        m.cols = self.rows
        m.data = m.__fill_data()
        return m

    def QRD(self):
        obj = lib.Matrix_QRD(self.obj)

        Q = Matrix.__new__(Matrix)
        Q.obj = lib.MatrixPair_A(obj)
        Q.rows = self.rows
        Q.cols = self.cols

        R = Matrix.__new__(Matrix)
        R.obj = lib.MatrixPair_B(obj)
        R.rows = self.rows
        R.cols = self.cols
        return Q, R
    
    def LUD(self):
        obj = lib.Matrix_LUD(self.obj)

        L = Matrix.__new__(Matrix)
        L.obj = lib.MatrixPair_A(obj)
        L.rows = self.rows
        L.cols = self.cols

        U = Matrix.__new__(Matrix)
        U.obj = lib.MatrixPair_B(obj)
        U.rows = self.rows
        U.cols = self.cols
        return L, U
    
    def SVD(self):
        obj = lib.Matrix_SVD(self.obj)

        U = Matrix.__new__(Matrix)
        U.obj = lib.MatrixTriple_A(obj)
        U.rows = self.rows
        U.cols = self.cols

        S = Matrix.__new__(Matrix)
        S.obj = lib.MatrixTriple_B(obj)
        S.rows = self.rows
        S.cols = self.cols

        V = Matrix.__new__(Matrix)
        V.obj = lib.MatrixTriple_C(obj)
        V.rows = self.rows
        V.cols = self.cols
        
        return U, S, V
    
    def eigenvalues(self):
        obj = lib.Matrix_Eigenvalues(self.obj)
        eigenvalues = []
        for i in range(self.rows):
            eigenvalues.append(lib.get_std_vector_element(obj, i))
        return eigenvalues
    
    def nullspace(self):
        obj = lib.Matrix_null_space(self.obj)
        count = lib.get_vector_count(obj)
        null_space = []
        for i in range(count):
            length = lib.get_vector_length(obj, i)
            vec_ptr = lib.get_std_vector_item(obj, i)
            null_space.append([lib.get_std_vector_element(vec_ptr, j) for j in range(length)])

        return null_space
    
    def eigenvectors(self):
        obj = lib.Matrix_Eigenvectors(self.obj)
        count = lib.get_vector_count(obj)
        eigenvectors = []
        for i in range(count):
            length = lib.get_vector_length(obj, i)
            vec_ptr = lib.get_std_vector_item(obj, i)
            eigenvectors.append([lib.get_std_vector_element(vec_ptr, j) for j in range(length)])

        return eigenvectors
    
    def rowreduce(self):
        obj = lib.Matrix_RREF(self.obj)
        result = Matrix.__new__(Matrix)
        result.obj = obj
        result.rows = self.rows
        result.cols = self.cols
        return result
    
    def determinant(self):
        if self.rows != self.cols:
            raise ValueError("Determinant can only be computed for square matrices.")
        return lib.Matrix_determinant(self.obj)
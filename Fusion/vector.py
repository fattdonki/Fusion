from .backend_bindings import lib
import ctypes
from .matrix import Matrix
import random

class Vector:
    def __init__(self, data):
        self.data = data
        self.size = len(data)
        arr = (ctypes.c_double * self.size)(*data)
        self.obj = lib.Vector_create(self.size, arr)

    def __del__(self):
        if hasattr(self, "obj"):
            lib.Vector_delete(self.obj)

    def __getitem__(self, index):
        if index < 0 or index >= self.size:
            raise IndexError("Index out of bounds.")
        
        return self.data[index]
    
    def __name__(self):
        return "Vector"
    
    def __str__(self):
        return "[" + ", ".join(str(v) for v in self.data) + "]"
    
    def __mul__(self, other):
        if isinstance(other, Vector):
            if self.size != other.size:
                raise ValueError("Vectors must have the same size for multiplication.")
        
            return lib.Vector_mul(self.obj, other.obj)
        
        elif isinstance(other, (int, float)):
            result_obj = lib.VecScalar_mul(self.obj, ctypes.c_double(other))
            result = Vector.__new__(Vector)
            result.obj = result_obj
            result.size = self.size
            result.data = result.__fill_data()
            return result
        
        elif isinstance(other, Matrix):
            if self.size != other.rows:
                raise ValueError("Matrix and vector dimensions do not match for multiplication.")
            
            result_obj = lib.VecMat_mul(self.obj, other.obj)
            result = Vector.__new__(Vector)
            result.obj = result_obj
            result.size = other.rows
            result.data = result.__fill_data()
            return result
        
        return NotImplemented        
        
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            result_obj = lib.VecScalar_mul(self.obj, ctypes.c_double(other))
            result = Vector.__new__(Vector)
            result.obj = result_obj
            result.size = self.size
            result.data = result.__fill_data()
            return result
 
        elif isinstance(other, Matrix):
            if self.size != other.rows:
                raise ValueError("Matrix and vector dimensions do not match for multiplication.")
            
            result_obj = lib.MatVec_mul(self.obj, other.obj)
            result = Vector.__new__(Vector)
            result.obj = result_obj
            result.size = other.rows
            result.data = result.__fill_data()
            return result

        return NotImplemented
        
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero is not allowed.")
            
            result_obj = lib.VecScalar_div(self.obj, ctypes.c_double(other))
            result = Vector.__new__(Vector)
            result.obj = result_obj
            result.size = self.size
            result.data = result.__fill_data()
            return result
        
        raise TypeError("Unsupported operand type(s) for /: 'Vector' and '{}'".format(type(other).__name__))
    
    def __add__(self, other):
        if self.size != other.size:
            raise ValueError("Vectors must have the same size for addition.")
        
        result_obj = lib.Vector_add(self.obj, other.obj)
        result = Vector.__new__(Vector)
        result.obj = result_obj
        result.size = self.size
        result.data = result.__fill_data()
        return result

    def __sub__(self, other):
        if self.size != other.size:
            raise ValueError("Vectors must have the same size for subtraction.")
        
        result_obj = lib.Vector_sub(self.obj, other.obj)
        result = Vector.__new__(Vector)
        result.obj = result_obj
        result.size = self.size
        result.data = result.__fill_data()
        return result
    
    def __eq__(self, other):
        return lib.Vector_eqq(self.obj, other.obj)
    
    def __ne__(self, other):
        return lib.Vector_neq(self.obj, other.obj)
    
    def __len__(self):
        return self.size
    
    def __fill_data(self):
        data = []
        for i in range(self.size):
            value = lib.Vector_find(self.obj, i)
            data.append(round(value,3))
        return data
    
    @staticmethod
    def zero(size):
        obj = lib.Vector_zero(size)
        v = Vector.__new__(Vector)
        v.obj = obj
        v.size = size
        v.data = [0.0] * size
        return v    
    
    @staticmethod
    def unit(size, axis=0):
        if axis < 0 or axis >= size:
            raise ValueError("Axis must be in the range [0, size-1].")
        
        obj = lib.Vector_unit(size, axis)
        v = Vector.__new__(Vector)
        v.obj = obj
        v.size = size
        v.data = v.__fill_data()
        return v
    
    @staticmethod
    def random(size, min_val=0.0, max_val=1.0):
        if min_val >= max_val:
            raise ValueError("min_val must be less than max_val.")
        
        data = [round(random.uniform(min_val, max_val), 2) for _ in range(size)]
        return Vector(data)
    
    def normalize(self):
        result_obj = lib.Vector_normalize(self.obj)
        result = Vector.__new__(Vector)
        result.obj = result_obj
        result.size = self.size
        result.data = result.__fill_data()
        return result
    
    def magnitude(self):
        return lib.Vector_magnitude(self.obj)
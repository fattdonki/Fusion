from .backend_bindings import lib
import ctypes

class Tensor:
    def __init__(self, shape, data):
        self.data = data
        self.shape = shape
        self.size = len(data)
        shape_arr = (ctypes.c_int * len(shape))(*shape)
        arr = (ctypes.c_double * self.size)(*data)
        self.obj = lib.Tensor_create(shape_arr, len(shape), arr)


    def __del__(self):
        if hasattr(self, "obj"):
            lib.Tensor_delete(self.obj)

    def __getitem__(self, index):
        if index >= self.size:
            raise IndexError("Index out of bounds.")
        
        return self.data[index]
    
    def __name__(self):
        return "Tensor"
    
    def __str__(self):
        def format_tensor(data, shape):
            if len(shape) == 1:
                return "[" + ", ".join(str(data[i]) for i in range(shape[0])) + "]"
            else:
                stride = int(len(data) / shape[0])
                return "[" + ",\n ".join(format_tensor(data[i*stride:(i+1)*stride], shape[1:]) for i in range(shape[0])) + "]"
        return format_tensor(self.data, self.shape)
    
    def __mul__(self, other):
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError("Tensor dimensions do not match for multiplication")
            
            result_obj = lib.Tensor_mul(self.obj, other.obj)
            result = Tensor.__new__(Tensor)
            result.obj = result_obj
            result.shape = self.shape + other.shape
            result.size = self.size * other.size
            result.data = result.__fill_data()
            return result

    def __fill_data(self):
        return [round(lib.Tensor_find(self.obj, i), 3) for i in range(self.size)]
import ctypes
import os

__here = os.path.dirname(os.path.abspath(__file__))

dll_path = os.path.abspath(
    os.path.join(__here, "..", "FusionBackend", "out", "build", "x64-debug", "FusionBackend.dll")
)
lib = ctypes.CDLL(dll_path)

##################################

sigmoid = lib.sigmoid
sigmoid.argtypes = [ctypes.c_double]
sigmoid.restype = ctypes.c_double

relu = lib.relu
relu.argtypes = [ctypes.c_double]
relu.restype = ctypes.c_double

factorial = lib.factorial
factorial.argtypes = [ctypes.c_int]
factorial.restype = ctypes.c_int

get_std_vector_element = lib.get_std_vector_element
get_std_vector_element.argtypes = [ctypes.c_void_p, ctypes.c_int]
get_std_vector_element.restype = ctypes.c_double

get_std_vector_item = lib.get_std_vector_item
get_std_vector_item.argtypes = [ctypes.c_void_p, ctypes.c_int]
get_std_vector_item.restype = ctypes.c_void_p

lib.get_vector_count = lib.get_vector_count
lib.get_vector_count.argtypes = [ctypes.c_void_p]
lib.get_vector_count.restype = ctypes.c_int

lib.get_vector_length = lib.get_vector_length
lib.get_vector_length.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.get_vector_length.restype = ctypes.c_int

##################################

lib.Matrix_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
lib.Matrix_create.restype = ctypes.c_void_p

lib.Matrix_delete.argtypes = [ctypes.c_void_p]
lib.Matrix_delete.restype = None

lib.Matrix_find.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.Matrix_find.restype = ctypes.c_double

lib.Matrix_identity.argtypes = [ctypes.c_int]
lib.Matrix_identity.restype = ctypes.c_void_p

lib.Matrix_zero.argtypes = [ctypes.c_int, ctypes.c_int]
lib.Matrix_zero.restype = ctypes.c_void_p

lib.Matrix_mul.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.Matrix_mul.restype = ctypes.c_void_p

lib.MatScalar_mul.argtypes = [ctypes.c_void_p, ctypes.c_double]
lib.MatScalar_mul.restype = ctypes.c_void_p

lib.Matrix_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.Matrix_add.restype = ctypes.c_void_p

lib.Matrix_sub.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.Matrix_sub.restype = ctypes.c_void_p

lib.Matrix_eqq.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.Matrix_eqq.restype = ctypes.c_bool

lib.Matrix_neq.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.Matrix_neq.restype = ctypes.c_bool

lib.Matrix_transpose.argtypes = [ctypes.c_void_p]
lib.Matrix_transpose.restype = ctypes.c_void_p

lib.Matrix_trace.argtypes = [ctypes.c_void_p]
lib.Matrix_trace.restype = ctypes.c_double

lib.Matrix_QRD.argtypes = [ctypes.c_void_p]
lib.Matrix_QRD.restype = ctypes.c_void_p 

lib.Matrix_LUD.argtypes = [ctypes.c_void_p]
lib.Matrix_LUD.restype = ctypes.c_void_p 

lib.Matrix_SVD.argtypes = [ctypes.c_void_p]
lib.Matrix_SVD.restype = ctypes.c_void_p 

lib.Matrix_Eigenvalues.argtypes = [ctypes.c_void_p]
lib.Matrix_Eigenvalues.restype = ctypes.c_void_p 

lib.Matrix_Eigenvectors.argtypes = [ctypes.c_void_p]
lib.Matrix_Eigenvectors.restype = ctypes.c_void_p 

lib.Matrix_null_space.argtypes = [ctypes.c_void_p]
lib.Matrix_null_space.restype = ctypes.c_void_p

lib.Matrix_RREF.argtypes = [ctypes.c_void_p]
lib.Matrix_RREF.restype = ctypes.c_void_p

lib.MatrixPair_A.argtypes = [ctypes.c_void_p]
lib.MatrixPair_A.restype = ctypes.c_void_p

lib.MatrixPair_B.argtypes = [ctypes.c_void_p]
lib.MatrixPair_B.restype = ctypes.c_void_p

lib.MatrixTriple_A.argtypes = [ctypes.c_void_p]
lib.MatrixTriple_A.restype = ctypes.c_void_p

lib.MatrixTriple_B.argtypes = [ctypes.c_void_p]
lib.MatrixTriple_B.restype = ctypes.c_void_p

lib.MatrixTriple_C.argtypes = [ctypes.c_void_p]
lib.MatrixTriple_C.restype = ctypes.c_void_p

lib.Matrix_determinant.argtypes = [ctypes.c_void_p]
lib.Matrix_determinant.restype = ctypes.c_double

##################################

lib.Vector_create.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
lib.Vector_create.restype = ctypes.c_void_p

lib.Vector_delete.argtypes = [ctypes.c_void_p]
lib.Vector_delete.restype = None

lib.Vector_find.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.Vector_find.restype = ctypes.c_double

lib.Vector_zero.argtypes = [ctypes.c_int]
lib.Vector_zero.restype = ctypes.c_void_p

lib.Vector_unit.argtypes = [ctypes.c_int, ctypes.c_int]
lib.Vector_unit.restype = ctypes.c_void_p

lib.Vector_mul.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.Vector_mul.restype = ctypes.c_double

lib.VecScalar_mul.argtypes = [ctypes.c_void_p, ctypes.c_double]
lib.VecScalar_mul.restype = ctypes.c_void_p

lib.VecScalar_div.argtypes = [ctypes.c_void_p, ctypes.c_double]
lib.VecScalar_div.restype = ctypes.c_void_p

lib.Vector_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.Vector_add.restype = ctypes.c_void_p

lib.Vector_sub.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.Vector_sub.restype = ctypes.c_void_p

lib.Vector_eqq.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.Vector_eqq.restype = ctypes.c_bool

lib.Vector_neq.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.Vector_neq.restype = ctypes.c_bool

lib.Vector_normalize.argtypes = [ctypes.c_void_p]
lib.Vector_normalize.restype = ctypes.c_void_p

lib.Vector_magnitude.argtypes = [ctypes.c_void_p]
lib.Vector_magnitude.restype = ctypes.c_double

##################################

lib.MatVec_mul.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.MatVec_mul.restype = ctypes.c_void_p

lib.VecMat_mul.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.VecMat_mul.restype = ctypes.c_void_p
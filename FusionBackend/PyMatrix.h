#include "Matrix.h"

PyObject* PyList_FromVector(const std::vector<double>& vec) {
    PyObject* list = PyList_New(vec.size());
    if (!list) return nullptr;

    for (size_t i = 0; i < vec.size(); ++i) {
        PyObject* num = PyFloat_FromDouble(vec[i]);
        if (!num) {
            Py_DECREF(list);
            return nullptr;
        }
        PyList_SET_ITEM(list, i, num);  
    }
    return list;
}

PyObject* PyList_FromVector(const std::vector<PyObject*>& vec) {
    PyObject* list = PyList_New(vec.size());
    if (!list) return nullptr;

    for (size_t i = 0; i < vec.size(); ++i) {
        PyObject* obj(vec[i]);
        if (!obj) {
            Py_DECREF(list);
            return nullptr;
        }
        PyList_SET_ITEM(list, i, obj);
    }
    return list;
}


struct PyMatrixObject {
    PyObject_HEAD Matrix* matrix;
};

static void PyMatrix_dealloc(PyMatrixObject* self) {
    delete self->matrix;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyMatrix_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PyMatrixObject* self = (PyMatrixObject*)type->tp_alloc(type, 0);
    if (!self) return nullptr;
    self->matrix = nullptr;
    return (PyObject*)self;
}

static int PyMatrix_init(PyMatrixObject* self, PyObject* args, PyObject* kwds) {
    PyObject* list_of_lists = nullptr;

    if (!PyArg_ParseTuple(args, "O", &list_of_lists))
        return -1;

    if (!PyList_Check(list_of_lists)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list of lists");
        return -1;
    }

    Py_ssize_t rows = PyList_Size(list_of_lists);
    if (rows == 0) {
        PyErr_SetString(PyExc_ValueError, "Empty list");
        return -1;
    }

    PyObject* first_row = PyList_GetItem(list_of_lists, 0);
    if (!PyList_Check(first_row)) {
        PyErr_SetString(PyExc_TypeError, "Expected nested lists");
        return -1;
    }

    Py_ssize_t cols = PyList_Size(first_row);
    std::vector<double> data;
    data.reserve(rows * cols);

    for (Py_ssize_t i = 0; i < rows; ++i) {
        PyObject* row = PyList_GetItem(list_of_lists, i);
        if (!PyList_Check(row) || PyList_Size(row) != cols) {
            PyErr_SetString(PyExc_ValueError, "Inconsistent row sizes");
            return -1;
        }

        for (Py_ssize_t j = 0; j < cols; ++j) {
            PyObject* item = PyList_GetItem(row, j);
            if (!PyFloat_Check(item) && !PyLong_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "Expected float or int");
                return -1;
            }
            data.push_back(PyFloat_AsDouble(item));
        }
    }

    delete self->matrix;
    self->matrix = new Matrix((int)rows, (int)cols, data.data());
    return 0;
}

static PyObject* PyMatrix_mul(PyObject* a, PyObject* b) {
    if (PyObject_TypeCheck(a, &PyMatrixType) && PyObject_TypeCheck(b, &PyMatrixType)) {
        Matrix* A = ((PyMatrixObject*)a)->matrix;
        Matrix* B = ((PyMatrixObject*)b)->matrix;
        if (!A || !B) {
            PyErr_SetString(PyExc_ValueError, "Uninitialized Matrix");
            return nullptr;
        }
        Matrix* result = new Matrix((*A) * (*B));
        PyMatrixObject* obj = PyObject_New(PyMatrixObject, &PyMatrixType);
        if (!obj) {
            delete result;
            return nullptr;
        }
        obj->matrix = result;
        return (PyObject*)obj;
    }

    if (PyObject_TypeCheck(a, &PyMatrixType) && (PyFloat_Check(b) || PyLong_Check(b))) {
        Matrix* A = ((PyMatrixObject*)a)->matrix;
        if (!A) {
            PyErr_SetString(PyExc_ValueError, "Uninitialized Matrix");
            return nullptr;
        }
        double scalar = PyFloat_AsDouble(b);
        Matrix* result = new Matrix((*A) * scalar);
        PyMatrixObject* obj = PyObject_New(PyMatrixObject, &PyMatrixType);
        if (!obj) {
            delete result;
            return nullptr;
        }
        obj->matrix = result;
        return (PyObject*)obj;
    }

    if ((PyFloat_Check(a) || PyLong_Check(a)) && PyObject_TypeCheck(b, &PyMatrixType)) {
        double scalar = PyFloat_AsDouble(a);
        Matrix* B = ((PyMatrixObject*)b)->matrix;
        if (!B) {
            PyErr_SetString(PyExc_ValueError, "Uninitialized Matrix");
            return nullptr;
        }
        Matrix* result = new Matrix((*B) * scalar);
        PyMatrixObject* obj = PyObject_New(PyMatrixObject, &PyMatrixType);
        if (!obj) {
            delete result;
            return nullptr;
        }
        obj->matrix = result;
        return (PyObject*)obj;
    }

    Py_RETURN_NOTIMPLEMENTED;
}


static PyObject* PyMatrix_add(PyObject* a, PyObject* b) {
    if (!PyObject_TypeCheck(a, &PyMatrixType) || !PyObject_TypeCheck(b, &PyMatrixType)) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    Matrix* A = ((PyMatrixObject*)a)->matrix;
    Matrix* B = ((PyMatrixObject*)b)->matrix;
    if (!A || !B) {
        PyErr_SetString(PyExc_ValueError, "Uninitialized Matrix");
        return nullptr;
    }

    Matrix* result = new Matrix((*A) + (*B));
    PyMatrixObject* obj = PyObject_New(PyMatrixObject, &PyMatrixType);
    if (!obj) {
        delete result;
        return nullptr;
    }

    obj->matrix = result;
    return (PyObject*)obj;
}

static PyObject* PyMatrix_sub(PyObject* a, PyObject* b) {
    if (!PyObject_TypeCheck(a, &PyMatrixType) || !PyObject_TypeCheck(b, &PyMatrixType)) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    Matrix* A = ((PyMatrixObject*)a)->matrix;
    Matrix* B = ((PyMatrixObject*)b)->matrix;
    if (!A || !B) {
        PyErr_SetString(PyExc_ValueError, "Uninitialized Matrix");
        return nullptr;
    }

    Matrix* result = new Matrix((*A) - (*B));
    PyMatrixObject* obj = PyObject_New(PyMatrixObject, &PyMatrixType);
    if (!obj) {
        delete result;
        return nullptr;
    }

    obj->matrix = result;
    return (PyObject*)obj;
}

static PyObject* PyMatrix_svd(PyMatrixObject* self, PyObject* Py_UNUSED(ignored)) {
    if (!self->matrix) {
        PyErr_SetString(PyExc_ValueError, "Uninitialized Matrix");
        return nullptr;
    }

    auto [U, S, V] = self->matrix->SVD();  

    PyMatrixObject* py_U_obj = PyObject_New(PyMatrixObject, &PyMatrixType);
    PyMatrixObject* py_S_obj = PyObject_New(PyMatrixObject, &PyMatrixType);
    PyMatrixObject* py_V_obj = PyObject_New(PyMatrixObject, &PyMatrixType);

    if (!py_U_obj || !py_S_obj || !py_V_obj) {
        if (py_U_obj && py_U_obj->matrix)
            delete py_U_obj->matrix;
        if (py_S_obj && py_S_obj->matrix)
            delete py_S_obj->matrix;
        if (py_V_obj && py_V_obj->matrix)
            delete py_V_obj->matrix;

        Py_XDECREF((PyObject*)py_U_obj);
        Py_XDECREF((PyObject*)py_S_obj);
        Py_XDECREF((PyObject*)py_V_obj);
        return nullptr;
    }

    py_U_obj->matrix = new Matrix(U);
    py_S_obj->matrix = new Matrix(S);
    py_V_obj->matrix = new Matrix(V);

    return PyTuple_Pack(3, (PyObject*)py_U_obj, (PyObject*)py_S_obj, (PyObject*)py_V_obj);

}

static PyObject* PyMatrix_qrd(PyMatrixObject* self, PyObject* Py_UNUSED(ignored)) {
    if (!self->matrix) {
        PyErr_SetString(PyExc_ValueError, "Uninitialized Matrix");
        return nullptr;
    }

    auto [Q, R] = self->matrix->QRD();

    PyMatrixObject* py_Q_obj = PyObject_New(PyMatrixObject, &PyMatrixType);
    PyMatrixObject* py_R_obj = PyObject_New(PyMatrixObject, &PyMatrixType);

    if (!py_Q_obj || !py_R_obj ) {
        if (py_Q_obj && py_Q_obj->matrix)
            delete py_Q_obj->matrix;
        if (py_R_obj && py_R_obj->matrix)
            delete py_R_obj->matrix;
        
        Py_XDECREF((PyObject*)py_Q_obj);
        Py_XDECREF((PyObject*)py_R_obj);
        return nullptr;
    }

    py_Q_obj->matrix = new Matrix(Q);
    py_R_obj->matrix = new Matrix(R);

    return PyTuple_Pack(2, (PyObject*)py_Q_obj, (PyObject*)py_R_obj);
}

static PyObject* PyMatrix_lud(PyMatrixObject* self, PyObject* Py_UNUSED(ignored)) {
    if (!self->matrix) {
        PyErr_SetString(PyExc_ValueError, "Uninitialized Matrix");
        return nullptr;
    }

    auto [L, U] = self->matrix->LUD();

    PyMatrixObject* py_L_obj = PyObject_New(PyMatrixObject, &PyMatrixType);
    PyMatrixObject* py_U_obj = PyObject_New(PyMatrixObject, &PyMatrixType);

    if (!py_L_obj || !py_U_obj) {
        if (py_L_obj && py_L_obj->matrix)
            delete py_L_obj->matrix;
        if (py_U_obj && py_U_obj->matrix)
            delete py_U_obj->matrix;

        Py_XDECREF((PyObject*)py_L_obj);
        Py_XDECREF((PyObject*)py_U_obj);
        return nullptr;
    }

    py_L_obj->matrix = new Matrix(L);
    py_U_obj->matrix = new Matrix(U);

    return PyTuple_Pack(2, (PyObject*)py_L_obj, (PyObject*)py_U_obj);

}

static PyObject* PyMatrix_eigenvalues(PyMatrixObject* self, PyObject* Py_UNUSED(ignored)) {
    if (!self->matrix) {
        PyErr_SetString(PyExc_ValueError, "Uninitialized Matrix");
        return nullptr;
    }
    return PyList_FromVector(self->matrix->eigenvalues());
}

static PyObject* PyMatrix_eigenvectors(PyMatrixObject* self, PyObject Py_UNUSED(ignored)) {
    if (!self->matrix) {
        PyErr_SetString(PyExc_ValueError, "Uninitialized Matrix");
        return nullptr;
    }
    std::vector<std::vector<double>> eigenvecs = self->matrix->eigenvectors();
    size_t size = eigenvecs.size();
    std::vector<PyObject*> pyEigenvecs(size);
    for (int i = 0; i < size; i++)
        pyEigenvecs[i] = PyList_FromVector(eigenvecs[i]);
    
    return PyList_FromVector(pyEigenvecs);
}

static PyObject* PyMatrix_null_space(PyMatrixObject* self, PyObject Py_UNUSED(ignored)) {
    if (!self->matrix) {
        PyErr_SetString(PyExc_ValueError, "Uninitialized Matrix");
        return nullptr;
    }
    std::vector<std::vector<double>> basis = self->matrix->null_space();
    size_t size = basis.size();
    std::vector<PyObject*> pyBasis(size);
    for (int i = 0; i < size; i++)
        pyBasis[i] = PyList_FromVector(basis[i]);

    return PyList_FromVector(pyBasis);
}

static PyObject* PyMatrix_RREF(PyMatrixObject* self, PyObject Py_UNUSED(ignored)) {
    if (!self->matrix) {
        PyErr_SetString(PyExc_ValueError, "Uninitialized Matrix");
        return nullptr;
    }

    Matrix* result = new Matrix(self->matrix->RREF());
    PyMatrixObject* obj = PyObject_New(PyMatrixObject, &PyMatrixType);
    if (!obj) {
        delete result;
        return nullptr;
    }
    obj->matrix = result;
    return (PyObject*)obj;
}

static PyObject* PyMatrix_richcompare(PyObject* a, PyObject* b, int op) {
    if (op != Py_EQ) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    if (!PyObject_TypeCheck(a, &PyMatrixType) || !PyObject_TypeCheck(b, &PyMatrixType)) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    Matrix* A = ((PyMatrixObject*)a)->matrix;
    Matrix* B = ((PyMatrixObject*)b)->matrix;

    if (!A || !B) {
        Py_RETURN_FALSE;
    }

    if (*A == *B) {
        Py_RETURN_TRUE;
    }
    else {
        Py_RETURN_FALSE;
    }
}

static PyObject* PyMatrix_trace(PyMatrixObject* self, PyObject* Py_UNUSED(ignored)) {
    if (!self->matrix) {
        PyErr_SetString(PyExc_ValueError, "Uninitialized Matrix");
        return nullptr;
    }
    
    return PyFloat_FromDouble(self->matrix->trace());
}

static PyObject* PyMatrix_determinant(PyMatrixObject* self, PyObject* Py_UNUSED(ignored)) {
    if (!self->matrix) {
        PyErr_SetString(PyExc_ValueError, "Uninitialized Matrix");
        return nullptr;
    }

    return PyFloat_FromDouble(self->matrix->determinant());
}

static PyObject* PyMatrix_transpose(PyMatrixObject* self, PyObject* Py_UNUSED(ignored)){
    if (!self->matrix) {
        PyErr_SetString(PyExc_ValueError, "Uninitialized Matrix");
        return nullptr;
    }

    Matrix* result = new Matrix(self->matrix->transpose());
    PyMatrixObject* obj = PyObject_New(PyMatrixObject, &PyMatrixType);

    if (!obj) {
        delete result;
        return nullptr;
    }
    obj->matrix = result;
    return (PyObject*)obj;
}

static PyObject* PyMatrix_str(PyObject* self) {
    PyMatrixObject* obj = (PyMatrixObject*)self;
    std::string s = obj->matrix->to_str(); 
    return PyUnicode_FromString(s.c_str());
}

static PyObject* PyMatrix_static_identity(PyObject*, PyObject* args) {
    int n;
    if (!PyArg_ParseTuple(args, "i", &n))
        return nullptr;

    Matrix* mat = new Matrix(n, n);
    for (int i = 0; i < n; ++i) mat->set(i, i, 1.0);
    PyMatrixObject* obj = PyObject_New(PyMatrixObject, &PyMatrixType);
    if (!obj) return nullptr;
    obj->matrix = mat;
    return (PyObject*)obj;
}

static PyObject* PyMatrix_static_zero(PyObject*, PyObject* args) {
    int n, h;
    if (!PyArg_ParseTuple(args, "ii", &n, &h))
        return nullptr;

    Matrix* mat = new Matrix(n, h);
    PyMatrixObject* obj = PyObject_New(PyMatrixObject, &PyMatrixType);
    if (!obj) return nullptr;
    obj->matrix = mat;
    return (PyObject*)obj;
}

static PyObject* PyMatrix_static_Rx(PyObject*, PyObject* args) {
    int angle;
    if (!PyArg_ParseTuple(args, "i", &angle))
        return nullptr;

    Matrix* mat = new Matrix{ Matrix::Rx(angle) };
    PyMatrixObject* obj = PyObject_New(PyMatrixObject, &PyMatrixType);
    if (!obj) return nullptr;
    obj->matrix = mat;
    return (PyObject*)obj;
}

static PyObject* PyMatrix_static_Ry(PyObject*, PyObject* args) {
    int angle;
    if (!PyArg_ParseTuple(args, "i", &angle))
        return nullptr;

    Matrix* mat = new Matrix{ Matrix::Ry(angle) };
    PyMatrixObject* obj = PyObject_New(PyMatrixObject, &PyMatrixType);
    if (!obj) return nullptr;
    obj->matrix = mat;
    return (PyObject*)obj;
}

static PyObject* PyMatrix_static_Rz(PyObject*, PyObject* args) {
    int angle;
    if (!PyArg_ParseTuple(args, "i", &angle))
        return nullptr;

    Matrix* mat = new Matrix{ Matrix::Rz(angle) };
    PyMatrixObject* obj = PyObject_New(PyMatrixObject, &PyMatrixType);
    if (!obj) return nullptr;
    obj->matrix = mat;
    return (PyObject*)obj;
}

static PyMethodDef PyMatrix_methods[] = {
    {"svd", (PyCFunction)PyMatrix_svd, METH_NOARGS, "Compute the singular value decomposition"},
    {"qrd", (PyCFunction)PyMatrix_qrd, METH_NOARGS, "Compute the QR decomposition"},
    {"lud", (PyCFunction)PyMatrix_lud, METH_NOARGS, "Compute the LU decomposition"},
    {"trace", (PyCFunction)PyMatrix_trace, METH_NOARGS, "Compute the trace of a matrix"},
    {"determinant", (PyCFunction)PyMatrix_determinant, METH_NOARGS, "Compute the determinant of a matrix"},
    {"transpose", (PyCFunction)PyMatrix_transpose, METH_NOARGS, "Transpose a matrix"},
    {"eigenvalues", (PyCFunction)PyMatrix_eigenvalues, METH_NOARGS, "Compute the eigenvalues of a matrix"},
    {"eigenvectors", (PyCFunction)PyMatrix_eigenvectors, METH_NOARGS, "Compute the eigenvectors of a matrix"},
    {"rref", (PyCFunction)PyMatrix_RREF, METH_NOARGS, "Compute the row reduction echlon form of a matrix"},
    {"nullspace", (PyCFunction)PyMatrix_null_space, METH_NOARGS, "Compute the null space of a matrix"},
    {"zero", (PyCFunction)PyMatrix_static_zero, METH_VARARGS | METH_STATIC, "Create a zero matrix with specified dimension"},
    {"identity", (PyCFunction)PyMatrix_static_identity, METH_VARARGS | METH_STATIC, "Create an identity matrix with specified dimension"},
    {"rx", (PyCFunction)PyMatrix_static_Rx, METH_VARARGS | METH_STATIC, "Create a matrix which rotates vectors around the x axis"},
    {"ry", (PyCFunction)PyMatrix_static_Ry, METH_VARARGS | METH_STATIC, "Create a matrix which rotates vectors around the y axis"},
    {"rz", (PyCFunction)PyMatrix_static_Rz, METH_VARARGS | METH_STATIC, "Create a matrix which rotates vectors around the z axis"},

    {nullptr, nullptr, 0, nullptr}
};

static PyNumberMethods PyMatrix_as_number = {
    .nb_add = PyMatrix_add,
    .nb_subtract = PyMatrix_sub,
    .nb_multiply = PyMatrix_mul,
    .nb_remainder = nullptr,
    .nb_divmod = nullptr,
    .nb_power = nullptr,
    .nb_negative = nullptr,
    .nb_positive = nullptr,
    .nb_absolute = nullptr,
    .nb_bool = nullptr,
    .nb_invert = nullptr,
    .nb_lshift = nullptr,
    .nb_rshift = nullptr,
    .nb_and = nullptr,
    .nb_xor = nullptr,
    .nb_or = nullptr,
    .nb_int = nullptr,
    .nb_reserved = nullptr,
    .nb_float = nullptr,
    // Python 3.5+ only
    .nb_inplace_add = nullptr,
    .nb_inplace_subtract = nullptr,
    .nb_inplace_multiply = nullptr,
};

PyTypeObject PyMatrixType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "Fusion.Matrix",                     // tp_name
    sizeof(PyMatrixObject),             // tp_basicsize
    0,                                  // tp_itemsize
    (destructor)PyMatrix_dealloc,       // tp_dealloc
    0,                                  // tp_vectorcall_offset (formerly tp_print)
    0,                                  // tp_getattr
    0,                                  // tp_setattr
    0,                                  // tp_as_async
    PyMatrix_str,                       // tp_repr
    &PyMatrix_as_number,                // tp_as_number
    0,                                  // tp_as_sequence
    0,                                  // tp_as_mapping
    0,                                  // tp_hash
    0,                                  // tp_call
    PyMatrix_str,                       // tp_str
    0,                                  // tp_getattro
    0,                                  // tp_setattro
    0,                                  // tp_as_buffer
    Py_TPFLAGS_DEFAULT,                 // tp_flags
    "Matrix objects",                   // tp_doc
    0,                                  // tp_traverse
    0,                                  // tp_clear
    PyMatrix_richcompare,               // tp_richcompare
    0,                                  // tp_weaklistoffset
    0,                                  // tp_iter
    0,                                  // tp_iternext
    PyMatrix_methods,                   // tp_methods
    0,                                  // tp_members
    0,                                  // tp_getset
    0,                                  // tp_base
    0,                                  // tp_dict
    0,                                  // tp_descr_get
    0,                                  // tp_descr_set
    0,                                  // tp_dictoffset
    (initproc)PyMatrix_init,            // tp_init
    0,                                  // tp_alloc
    PyMatrix_new,                       // tp_new
    0,                                  // tp_free
    0,                                  // tp_is_gc
    0,                                  // tp_bases
    0,                                  // tp_mro
    0,                                  // tp_cache
    0,                                  // tp_subclasses
    0,                                  // tp_weaklist
    0,                                  // tp_del
    0,                                  // tp_version_tag
    0,                                  // tp_finalize
#if PY_VERSION_HEX >= 0x03080000
    0,                                  // tp_vectorcall
    0                                   // tp_print (removed long ago but still declared in headers)
#endif
};
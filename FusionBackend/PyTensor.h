#include "Tensor.h"

struct PyTensorObject {
    PyObject_HEAD Tensor* tensor;
};

static void PyTensor_dealloc(PyTensorObject* self) {
    delete self->tensor;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyTensor_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PyTensorObject* self = (PyTensorObject*)type->tp_alloc(type, 0);
    if (!self) return nullptr;
    self->tensor = nullptr;
    return (PyObject*)self;
}

static int PyTensor_init(PyTensorObject* self, PyObject* args, PyObject* kwds) {
    PyObject* shape_obj = nullptr;
    PyObject* values_obj = nullptr;

    if (!PyArg_ParseTuple(args, "O|O", &shape_obj, &values_obj))
        return -1;

    if (!PyList_Check(shape_obj)) {
        PyErr_SetString(PyExc_TypeError, "Shape must be a list of integers");
        return -1;
    }

    std::vector<int> shape;
    Py_ssize_t ndim = PyList_Size(shape_obj);
    shape.reserve(ndim);

    for (Py_ssize_t i = 0; i < ndim; ++i) {
        PyObject* item = PyList_GetItem(shape_obj, i);
        if (!PyLong_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "Shape elements must be integers");
            return -1;
        }
        shape.push_back(static_cast<int>(PyLong_AsLong(item)));
    }

    int total_size = 1;
    for (int dim : shape)
        total_size *= dim;

    std::vector<double> values;
    const double* raw_data = nullptr;

    if (values_obj && values_obj != Py_None) {
        if (!PyList_Check(values_obj)) {
            PyErr_SetString(PyExc_TypeError, "Values must be a list of floats");
            return -1;
        }

        if (PyList_Size(values_obj) != total_size) {
            PyErr_SetString(PyExc_ValueError, "Values list size does not match total tensor size");
            return -1;
        }

        values.reserve(total_size);
        for (Py_ssize_t i = 0; i < total_size; ++i) {
            PyObject* item = PyList_GetItem(values_obj, i);
            if (!PyFloat_Check(item) && !PyLong_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "Values must be float or int");
                return -1;
            }
            values.push_back(PyFloat_AsDouble(item));
        }
        raw_data = values.data();
    }

    if (self->tensor) delete self->tensor;
    self->tensor = new Tensor(shape, raw_data);
    return 0;
}


static PyObject* PyTensor_mul(PyObject* a, PyObject* b) {
    if (PyObject_TypeCheck(a, &PyTensorType) && PyObject_TypeCheck(b, &PyTensorType)) {
        Tensor* A = ((PyTensorObject*)a)->tensor;
        Tensor* B = ((PyTensorObject*)b)->tensor;
        if (!A || !B) {
            PyErr_SetString(PyExc_ValueError, "Uninitialized Matrix");
            return nullptr;
        }
        Tensor* result = new Tensor((*A) * (*B));
        PyTensorObject* obj = PyObject_New(PyTensorObject, &PyTensorType);
        if (!obj) {
            delete result;
            return nullptr;
        }
        obj->tensor = result;
        return (PyObject*)obj;
    }

    if (PyObject_TypeCheck(a, &PyTensorType) && (PyFloat_Check(b) || PyLong_Check(b))) {
        Tensor* A = ((PyTensorObject*)a)->tensor;
        if (!A) {
            PyErr_SetString(PyExc_ValueError, "Uninitialized Tensor");
            return nullptr;
        }
        double scalar = PyFloat_AsDouble(b);
        Tensor* result = new Tensor((*A) * scalar);
        PyTensorObject* obj = PyObject_New(PyTensorObject, &PyTensorType);
        if (!obj) {
            delete result;
            return nullptr;
        }
        obj->tensor = result;
        return (PyObject*)obj;
    }

    if ((PyFloat_Check(a) || PyLong_Check(a)) && PyObject_TypeCheck(b, &PyTensorType)) {
        double scalar = PyFloat_AsDouble(a);
        Tensor* B = ((PyTensorObject*)b)->tensor;
        if (!B) {
            PyErr_SetString(PyExc_ValueError, "Uninitialized Tensor");
            return nullptr;
        }
        Tensor* result = new Tensor((*B) * scalar);
        PyTensorObject* obj = PyObject_New(PyTensorObject, &PyTensorType);
        if (!obj) {
            delete result;
            return nullptr;
        }
        obj->tensor = result;
        return (PyObject*)obj;
    }

    Py_RETURN_NOTIMPLEMENTED;
}


static PyObject* PyTensor_add(PyObject* a, PyObject* b) {
    if (PyObject_TypeCheck(a, &PyTensorType) || PyObject_TypeCheck(b, &PyTensorType)) {
        Tensor* A = ((PyTensorObject*)a)->tensor;
        Tensor* B = ((PyTensorObject*)b)->tensor;
        if (!A || !B) {
            PyErr_SetString(PyExc_ValueError, "Uninitialized Tensor");
            return nullptr;
        }

        Tensor* result = new Tensor((*A) + (*B));
        PyTensorObject* obj = PyObject_New(PyTensorObject, &PyTensorType);
        if (!obj) {
            delete result;
            return nullptr;
        }

        obj->tensor = result;
        return (PyObject*)obj;
    }
    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject* PyTensor_div(PyObject* a, PyObject* b) {
    if (PyObject_TypeCheck(a, &PyTensorType) && (PyFloat_Check(b) || PyLong_Check(b))) {
        Tensor* A = ((PyTensorObject*)a)->tensor;
        if (!A) {
            PyErr_SetString(PyExc_ValueError, "Uninitialized Tensor");
            return nullptr;
        }
        double scalar = PyFloat_AsDouble(b);
        Tensor* result = new Tensor((*A) / scalar);
        PyTensorObject* obj = PyObject_New(PyTensorObject, &PyTensorType);
        if (!obj) {
            delete result;
            return nullptr;
        }
        obj->tensor = result;
        return (PyObject*)obj;
    }
    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject* PyTensor_sub(PyObject* a, PyObject* b) {
    if (PyObject_TypeCheck(a, &PyTensorType) || PyObject_TypeCheck(b, &PyTensorType)) {

        Tensor* A = ((PyTensorObject*)a)->tensor;
        Tensor* B = ((PyTensorObject*)b)->tensor;
        if (!A || !B) {
            PyErr_SetString(PyExc_ValueError, "Uninitialized Tensor");
            return nullptr;
        }

        Tensor* result = new Tensor((*A) - (*B));
        PyTensorObject* obj = PyObject_New(PyTensorObject, &PyTensorType);
        if (!obj) {
            delete result;
            return nullptr;
        }

        obj->tensor = result;
        return (PyObject*)obj;
    }
    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject* PyTensor_str(PyObject* self) {
    PyTensorObject* obj = (PyTensorObject*)self;
    std::string s = obj->tensor->to_str();
    return PyUnicode_FromString(s.c_str());
}

static PyObject* PyTensor_richcompare(PyObject* a, PyObject* b, int op) {
    if (op != Py_EQ) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    if (!PyObject_TypeCheck(a, &PyTensorType) || !PyObject_TypeCheck(b, &PyTensorType)) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    Tensor* A = ((PyTensorObject*)a)->tensor;
    Tensor* B = ((PyTensorObject*)b)->tensor;

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

static PyMethodDef PyTensor_methods[] = {
    {nullptr, nullptr, 0, nullptr}
};

static PyNumberMethods PyTensor_as_number = {
    .nb_add = PyTensor_add,
    .nb_subtract = PyTensor_sub,
    .nb_multiply = PyTensor_mul,
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
    .nb_true_divide = PyTensor_div,
};

PyTypeObject PyTensorType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "Fusion.Tensor",                     // tp_name
    sizeof(PyTensorObject),             // tp_basicsize
    0,                                  // tp_itemsize
    (destructor)PyTensor_dealloc,       // tp_dealloc
    0,                                  // tp_vectorcall_offset (formerly tp_print)
    0,                                  // tp_getattr
    0,                                  // tp_setattr
    0,                                  // tp_as_async
    PyTensor_str,                       // tp_repr
    &PyTensor_as_number,                // tp_as_number
    0,                                  // tp_as_sequence
    0,                                  // tp_as_mapping
    0,                                  // tp_hash
    0,                                  // tp_call
    PyTensor_str,                       // tp_str
    0,                                  // tp_getattro
    0,                                  // tp_setattro
    0,                                  // tp_as_buffer
    Py_TPFLAGS_DEFAULT,                 // tp_flags
    "Tensor objects",                   // tp_doc
    0,                                  // tp_traverse
    0,                                  // tp_clear
    PyTensor_richcompare,               // tp_richcompare
    0,                                  // tp_weaklistoffset
    0,                                  // tp_iter
    0,                                  // tp_iternext
    PyTensor_methods,                   // tp_methods
    0,                                  // tp_members
    0,                                  // tp_getset
    0,                                  // tp_base
    0,                                  // tp_dict
    0,                                  // tp_descr_get
    0,                                  // tp_descr_set
    0,                                  // tp_dictoffset
    (initproc)PyTensor_init,            // tp_init
    0,                                  // tp_alloc
    PyTensor_new,                       // tp_new
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
#include "Vector.h"

struct PyVectorObject {
    PyObject_HEAD Vector* vector;
};

static void PyVector_dealloc(PyVectorObject* self) {
    delete self->vector;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyVector_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PyVectorObject* self = (PyVectorObject*)type->tp_alloc(type, 0);
    if (!self) return nullptr;
    self->vector = nullptr;
    return (PyObject*)self;
}

static int PyVector_init(PyVectorObject* self, PyObject* args, PyObject* kwds) {
    PyObject* list = nullptr;

    if (!PyArg_ParseTuple(args, "O", &list))
        return -1;

    if (!PyList_Check(list)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list");
        return -1;
    }

    Py_ssize_t size = PyList_Size(list);
    if (size == 0) {
        PyErr_SetString(PyExc_ValueError, "Empty list");
        return -1;
    }

    std::vector<double> data;
    data.reserve(size);

    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject* item = PyList_GetItem(list, i);
        if (!PyFloat_Check(item) && !PyLong_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "Expected float or int");
            return -1;
        }
        data.push_back(PyFloat_AsDouble(item));
    }

    delete self->vector;
    self->vector = new Vector((int)size, data.data());
    return 0;
}

static PyObject* PyVector_mul(PyObject* a, PyObject* b) {
    if (PyObject_TypeCheck(a, &PyVectorType) && PyObject_TypeCheck(b, &PyVectorType)) {
        Vector* A = ((PyVectorObject*)a)->vector;
        Vector* B = ((PyVectorObject*)b)->vector;
        if (!A || !B) {
            PyErr_SetString(PyExc_ValueError, "Uninitialized Vector");
            return nullptr;
        }
        return PyFloat_FromDouble((*A)*(*B));
    }

    if (PyObject_TypeCheck(a, &PyVectorType) && PyObject_TypeCheck(b, &PyMatrixType)) {
        Vector* A = ((PyVectorObject*)a)->vector;
        Matrix* B = ((PyMatrixObject*)b)->matrix;
        if (!A) {
            PyErr_SetString(PyExc_ValueError, "Uninitialized Vector");
            return nullptr;
        }
        if (!B) {
            PyErr_SetString(PyExc_ValueError, "Uninitialized Matrix");
            return nullptr;
        }
        Vector* result = new Vector((*A) * (*B));
        PyVectorObject* obj = PyObject_New(PyVectorObject, &PyVectorType);
        if (!obj) {
            delete result;
            return nullptr;
        }
        obj->vector = result;
        return (PyObject*)obj;
    }

    if (PyObject_TypeCheck(a, &PyMatrixType) && PyObject_TypeCheck(b, &PyVectorType)) {
        Matrix* A = ((PyMatrixObject*)a)->matrix;
        Vector* B = ((PyVectorObject*)b)->vector;
        if (!A) {
            PyErr_SetString(PyExc_ValueError, "Uninitialized Vector");
            return nullptr;
        }
        if (!B) {
            PyErr_SetString(PyExc_ValueError, "Uninitialized Matrix");
            return nullptr;
        }
        Vector* result = new Vector((*A) * (*B));
        PyVectorObject* obj = PyObject_New(PyVectorObject, &PyVectorType);
        if (!obj) {
            delete result;
            return nullptr;
        }
        obj->vector = result;
        return (PyObject*)obj;
    }

    if (PyObject_TypeCheck(a, &PyVectorType) && (PyFloat_Check(b) || PyLong_Check(b))) {
        Vector* A = ((PyVectorObject*)a)->vector;
        if (!A) {
            PyErr_SetString(PyExc_ValueError, "Uninitialized Vector");
            return nullptr;
        }
        double scalar = PyFloat_AsDouble(b);
        Vector* result = new Vector((*A) * scalar);
        PyVectorObject* obj = PyObject_New(PyVectorObject, &PyVectorType);
        if (!obj) {
            delete result;
            return nullptr;
        }
        obj->vector = result;
        return (PyObject*)obj;
    }

    if ((PyFloat_Check(a) || PyLong_Check(a)) && PyObject_TypeCheck(b, &PyVectorType)) {
        double scalar = PyFloat_AsDouble(a);
        Vector* B = ((PyVectorObject*)b)->vector;
        if (!B) {
            PyErr_SetString(PyExc_ValueError, "Uninitialized Vector");
            return nullptr;
        }
        Vector* result = new Vector((*B) * scalar);
        PyVectorObject* obj = PyObject_New(PyVectorObject, &PyVectorType);
        if (!obj) {
            delete result;
            return nullptr;
        }
        obj->vector = result;
        return (PyObject*)obj;
    }

    Py_RETURN_NOTIMPLEMENTED;
}


static PyObject* PyVector_add(PyObject* a, PyObject* b) {
    if (PyObject_TypeCheck(a, &PyVectorType) || PyObject_TypeCheck(b, &PyVectorType)) {
        Vector* A = ((PyVectorObject*)a)->vector;
        Vector* B = ((PyVectorObject*)b)->vector;
        if (!A || !B) {
            PyErr_SetString(PyExc_ValueError, "Uninitialized Vector");
            return nullptr;
        }

        Vector* result = new Vector((*A) + (*B));
        PyVectorObject* obj = PyObject_New(PyVectorObject, &PyVectorType);
        if (!obj) {
            delete result;
            return nullptr;
        }

        obj->vector = result;
        return (PyObject*)obj;
    }
    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject* PyVector_div(PyObject* a, PyObject* b) {
    if (PyObject_TypeCheck(a, &PyVectorType) && (PyFloat_Check(b) || PyLong_Check(b))) {
        Vector* A = ((PyVectorObject*)a)->vector;
        if (!A) {
            PyErr_SetString(PyExc_ValueError, "Uninitialized Vector");
            return nullptr;
        }
        double scalar = PyFloat_AsDouble(b);
        Vector* result = new Vector((*A) / scalar);
        PyVectorObject* obj = PyObject_New(PyVectorObject, &PyVectorType);
        if (!obj) {
            delete result;
            return nullptr;
        }
        obj->vector = result;
        return (PyObject*)obj;
    }
    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject* PyVector_sub(PyObject* a, PyObject* b) {
    if (PyObject_TypeCheck(a, &PyVectorType) || PyObject_TypeCheck(b, &PyVectorType)) {

        Vector* A = ((PyVectorObject*)a)->vector;
        Vector* B = ((PyVectorObject*)b)->vector;
        if (!A || !B) {
            PyErr_SetString(PyExc_ValueError, "Uninitialized Vector");
            return nullptr;
        }

        Vector* result = new Vector((*A) - (*B));
        PyVectorObject* obj = PyObject_New(PyVectorObject, &PyVectorType);
        if (!obj) {
            delete result;
            return nullptr;
        }

        obj->vector = result;
        return (PyObject*)obj;
    }
    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject* PyVector_str(PyObject* self) {
    PyVectorObject* obj = (PyVectorObject*)self;
    std::string s = obj->vector->to_str();
    return PyUnicode_FromString(s.c_str());
}

static PyObject* PyVector_richcompare(PyObject* a, PyObject* b, int op) {
    if (op != Py_EQ) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    if (!PyObject_TypeCheck(a, &PyVectorType) || !PyObject_TypeCheck(b, &PyVectorType)) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    Vector* A = ((PyVectorObject*)a)->vector;
    Vector* B = ((PyVectorObject*)b)->vector;

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

static PyObject* PyVector_normalize(PyVectorObject* self, PyObject* Py_UNUSED(ignored)) {
    if (!self->vector) {
        PyErr_SetString(PyExc_ValueError, "Uninitialized Matrix");
        return nullptr;
    }

    Vector* result = new Vector(self->vector->normalize());
    PyVectorObject* obj = PyObject_New(PyVectorObject, &PyVectorType);

    if (!obj) {
        delete result;
        return nullptr;
    }
    obj->vector = result;
    return (PyObject*)obj;
}

static PyObject* PyVector_magnitude(PyVectorObject* self, PyObject* Py_UNUSED(ignored)) {
    if (!self->vector) {
        PyErr_SetString(PyExc_ValueError, "Uninitialized Matrix");
        return nullptr;
    }

    return PyFloat_FromDouble(self->vector->magnitude());
}

static PyMethodDef PyVector_methods[] = {
    {"normalize", (PyCFunction)PyVector_normalize, METH_NOARGS, "Normalize a vector"},
    {"magnitude", (PyCFunction)PyVector_magnitude, METH_NOARGS, "Compute the magnitude of a vector"},

    {nullptr, nullptr, 0, nullptr}
};

static PyNumberMethods PyVector_as_number = {
    .nb_add = PyVector_add,
    .nb_subtract = PyVector_sub,
    .nb_multiply = PyVector_mul,
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
    .nb_true_divide = PyVector_div,
};

PyTypeObject PyVectorType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "Fusion.Vector",                     // tp_name
    sizeof(PyVectorObject),             // tp_basicsize
    0,                                  // tp_itemsize
    (destructor)PyVector_dealloc,       // tp_dealloc
    0,                                  // tp_vectorcall_offset (formerly tp_print)
    0,                                  // tp_getattr
    0,                                  // tp_setattr
    0,                                  // tp_as_async
    PyVector_str,                       // tp_repr
    &PyVector_as_number,                // tp_as_number
    0,                                  // tp_as_sequence
    0,                                  // tp_as_mapping
    0,                                  // tp_hash
    0,                                  // tp_call
    PyVector_str,                       // tp_str
    0,                                  // tp_getattro
    0,                                  // tp_setattro
    0,                                  // tp_as_buffer
    Py_TPFLAGS_DEFAULT,                 // tp_flags
    "Vector objects",                   // tp_doc
    0,                                  // tp_traverse
    0,                                  // tp_clear
    PyVector_richcompare,               // tp_richcompare
    0,                                  // tp_weaklistoffset
    0,                                  // tp_iter
    0,                                  // tp_iternext
    PyVector_methods,                   // tp_methods
    0,                                  // tp_members
    0,                                  // tp_getset
    0,                                  // tp_base
    0,                                  // tp_dict
    0,                                  // tp_descr_get
    0,                                  // tp_descr_set
    0,                                  // tp_dictoffset
    (initproc)PyVector_init,            // tp_init
    0,                                  // tp_alloc
    PyVector_new,                       // tp_new
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
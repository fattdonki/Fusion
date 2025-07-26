#include "PyMatrix.h"
#include "PyFunctions.h"
#include "PyVector.h"
#include "PyTensor.h"

static PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "Fusion",
    nullptr,
    -1,
    Fusion_methods
};

extern "C" __declspec(dllexport) PyObject* PyInit_Fusion(void) {
    if (PyType_Ready(&PyMatrixType) < 0)
        return nullptr;

    if (PyType_Ready(&PyVectorType) < 0)
        return nullptr;

    if (PyType_Ready(&PyTensorType) < 0)
        return nullptr;

    PyObject* m = PyModule_Create(&moduledef);
    if (!m) return nullptr;

    Py_INCREF(&PyMatrixType);
    if (PyModule_AddObject(m, "Matrix", (PyObject*)&PyMatrixType) < 0) {
        Py_DECREF(&PyMatrixType);
        Py_DECREF(m);
        return nullptr;
    }

    Py_INCREF(&PyVectorType);
    if (PyModule_AddObject(m, "Vector", (PyObject*)&PyVectorType) < 0) {
        Py_DECREF(&PyVectorType);
        Py_DECREF(&PyMatrixType);
        Py_DECREF(m);
        return nullptr;
    }

    Py_INCREF(&PyTensorType);
    if (PyModule_AddObject(m, "Tensor", (PyObject*)&PyTensorType) < 0) {
        Py_DECREF(&PyTensorType);
        Py_DECREF(&PyVectorType);
        Py_DECREF(&PyMatrixType);
        Py_DECREF(m);
        return nullptr;
    }

    return m;
}

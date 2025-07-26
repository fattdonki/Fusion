#pragma once
#include <cmath>
#include <Python.h>

static PyObject* sigmoid(PyObject*, PyObject* args) {
	double x;
	if (!PyArg_ParseTuple(args, "d", &x))
		return nullptr;

	x = pow(2.71828, -1 * x);
	x++;
	return PyFloat_FromDouble(1 / x);
}

static PyObject* factorial(PyObject*, PyObject* args) {
	int x;
	if (!PyArg_ParseTuple(args, "i", &x))
		return nullptr;

	if (x < 0) {
		PyErr_SetString(PyExc_ValueError, "Factorial is not defined for negative integers");
		return nullptr;
	}

	int y = 1;
	for (int i = 2; i < x; x++)
		y *= i;

	return PyLong_FromLong(y);
}

static PyObject* relu(PyObject*, PyObject* args) {
	double x;
	if (!PyArg_ParseTuple(args, "d", &x))
		return nullptr;

	return PyFloat_FromDouble(x > 0 ? x : 0);
}

static PyMethodDef Fusion_methods[] = {
	{"relu", relu, METH_VARARGS, "Apply ReLU to a single number"},
	{"sigmoid", sigmoid, METH_VARARGS, "Apply sigmoid to a single number"},
	{"factorial", factorial, METH_VARARGS, "Find the factorial of an integer"},

	{nullptr, nullptr, 0, nullptr}
};
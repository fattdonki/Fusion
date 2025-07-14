#pragma once
#include <cmath>
#include <vector>

extern "C" __declspec(dllexport) double sigmoid(double x) {
	x = pow(2.71828, -1 * x);
	x++;
	return 1 / x;
}

extern "C" __declspec(dllexport) int factorial(int x) {
	int y = x;
	for (int i = 2; i < y; i++) {
		x *= i;
	}
	return x;
}

extern "C" __declspec(dllexport) double relu(double x) {
	return x > 0 ? x : 0;
}

extern "C" __declspec(dllexport) double get_std_vector_element(void* vec, int index) {
	std::vector<double>* V = static_cast<std::vector<double>*>(vec);
	return V->at(index); 
}

extern "C" __declspec(dllexport) void* get_std_vector_item(void* vec, int index) {
	std::vector<std::vector<double>>* V = static_cast<std::vector<std::vector<double>>*>(vec);
	return new std::vector<double>(V->at(index));
}

extern "C" __declspec(dllexport) int get_vector_count(void* vec) {
	return static_cast<int>(static_cast<std::vector<std::vector<double>>*>(vec)->size());
}

extern "C" __declspec(dllexport) int get_vector_length(void* vec, int i) {
	return static_cast<int>(static_cast<std::vector<std::vector<double>>*>(vec)->at(i).size());
}
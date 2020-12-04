#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>

using namespace std;

uint32_t wgtPick(vector<double> wgt) {
	uint32_t n = wgt.size();
	int i;

	double sum = 0;
	for (i = 0; i < n; i++) {
		sum += wgt[i];
	}

	for (i = 0; i < n; i++) {
		wgt[i] /= sum;
	}

	double ratio = ((double)rand()) / RAND_MAX;
	for (i = 0; i < n; i++) {
		ratio -= wgt[i];
		if (ratio <= 0) {
			return i;
		}
	}

	return n - 1;
}

uint32_t softmaxPick(double rating, vector<double> ratings) {
	uint32_t n = ratings.size();
	int i;

	vector<double> wgt(n);
	for (i = 0; i < n; i++) {
		wgt[i] = exp(-abs(ratings[i] - rating));
	}

	return wgtPick(wgt);
}

template<typename T>
T UniformPick(vector<T> pool) {
	uint32_t n = pool.size();
	return pool[(uint8_t) rand() % n];
}
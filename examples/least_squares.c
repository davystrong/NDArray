#include <stdlib.h>
#include <stdio.h>
#include "ndarray.h"
#include <pthread.h>
#include <math.h>

struct NDArray *createA(float *temps, float *volts, int n, int k)
{
    int shape[] = {n, k + 1};
    struct NDArray *X = NDArray_zeros(shape, 2);
    shape[1] = 1;
    struct NDArray *y = NDArray_zeros(shape, 2);

    int index[2];
    for (int i = 0; i < n; i++)
    {
        index[0] = i;
        index[1] = 0;
        NDArray_set(y, index, temps[i]);

        for (int j = 0; j <= k; j++)
        {
            index[1] = j;
            NDArray_set(X, index, powf(volts[i], j));
        }
    }

    NDArray_print(X);
    NDArray_print(y);

    struct NDArray *Xt = NDArray_copy(X);
    NDArray_swapAxes(Xt, 0, 1);

    struct NDArray *prod1 = NDArray_matmul(Xt, X);
    NDArray_free(X);

    struct NDArray *inv = NDArray_inv(prod1);
    NDArray_free(prod1);

    struct NDArray *prod2 = NDArray_matmul(inv, Xt);
    NDArray_free(inv);
    NDArray_free(Xt);

    struct NDArray *a = NDArray_matmul(prod2, y);
    NDArray_free(prod2);
    NDArray_free(y);
    return a;
}

float getTemp(struct NDArray *a, float x)
{
    int shape[2];
    shape[0] = 1;
    shape[1] = a->shape[0];
    struct NDArray *X = NDArray_zeros(shape, 2);

    int index[2];
    index[0] = 0;
    for (int j = 0; j < a->shape[0]; j++)
    {
        index[1] = j;
        NDArray_set(X, index, powf(x, j));
    }

    struct NDArray *y = NDArray_matmul(X, a);
    float output = y->data[0];

    NDArray_free(X);
    NDArray_free(y);

    return output;
}

int main()
{
    // pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);

    int k = 3;
    int n = 5;

    float temps[] = {61.00, 56.10, 51.70, 45.50, 43.10};
    float volts[] = {2.120, 2.340, 2.540, 2.850, 2.960};
    float testVal = 2.850;
    float output;
    struct NDArray *a;

    a = createA(temps, volts, n, k);
    for (int i = 0; i < 2; i++)
    {
        // Testing
        output = getTemp(a, testVal);
        printf("%4.4f\n", output);
    }
    NDArray_free(a);
}
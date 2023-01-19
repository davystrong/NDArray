#include <stdlib.h>
#include <stdio.h>
#include "ndarray.h"
#include <pthread.h>

int main()
{
    // pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    // pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0);

    int shape[] = {4, 4, 0, 0};
    int index[] = {0, 0, 0, 0};
    int order[] = {0, 1, 2, 3};
    int errorCode;

    struct NDArray *array = NDArray_zeros(shape, 2);

    NDArray_set(array, index, 1.0);
    index[0] = 1;
    NDArray_set(array, index, 1.0);
    index[1] = 1;
    NDArray_set(array, index, 1.0);
    index[0] = 0;
    NDArray_set(array, index, 1.0);

    index[0] = 2;
    index[1] = 2;
    NDArray_set(array, index, 1.0);
    index[0] = 3;
    NDArray_set(array, index, 1.0);
    index[1] = 3;
    NDArray_set(array, index, 1.0);
    index[0] = 2;
    NDArray_set(array, index, 1.0);

    shape[0] = -1;
    shape[1] = 2;
    shape[2] = 2;
    shape[3] = 2;
    errorCode = NDArray_reshape(array, shape, 4);
    if (errorCode)
    {
        printf("Reshape failed with error code %d\n", errorCode);
        return 1;
    }

    order[0] = 0;
    order[1] = 2;
    order[2] = 1;
    order[3] = 3;
    errorCode = NDArray_transpose(array, order);
    if (errorCode)
    {
        printf("Transpose failed with error code %d\n", errorCode);
        return 1;
    }

    shape[0] = -1;
    shape[1] = 4;
    errorCode = NDArray_reshape(array, shape, 2);
    if (errorCode)
    {
        printf("Reshape failed with error code %d\n", errorCode);
        return 1;
    }

    NDArray_print(array);

    struct NDArray *summedArray = NDArray_sum(array, 1);

    shape[0] = 1;
    shape[1] = -1;
    errorCode = NDArray_reshape(summedArray, shape, 2);
    if (errorCode)
    {
        printf("Reshape failed with error code %d\n", errorCode);
        return 1;
    }

    NDArray_print(summedArray);

    struct NDArray *multipliedArray = NDArray_multiply(array, summedArray);
    NDArray_print(multipliedArray);

    struct NDArray *matmulledArray = NDArray_matmul(summedArray, multipliedArray);

    NDArray_print(matmulledArray);

    NDArray_free(matmulledArray);
    NDArray_free(multipliedArray);
    NDArray_free(summedArray);
    NDArray_free(array);

    printf("Testing invert...\n");

    struct NDArray *eye = NDArray_eye(5);
    struct NDArray *minusOne = NDArray_single(-1, 2);
    struct NDArray *negNotEye = NDArray_add(eye, minusOne);
    NDArray_free(eye);
    struct NDArray *notEye = NDArray_multiply(negNotEye, minusOne);
    NDArray_free(minusOne);
    NDArray_free(negNotEye);
    index[0] = 2;
    index[1] = 3;
    NDArray_set(notEye, index, 7);
    struct NDArray *inv = NDArray_inv(notEye);

    printf("\nInv input: ");
    NDArray_print(notEye);

    printf("\nInv output: ");
    NDArray_print(inv);

    NDArray_free(notEye);
    NDArray_free(inv);
}
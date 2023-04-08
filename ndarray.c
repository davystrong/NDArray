#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "ndarray.h"
#include <stdbool.h>

#ifdef DEBUG
#define DEBUG_PRINT(...)              \
    do                                \
    {                                 \
        fprintf(stderr, __VA_ARGS__); \
    } while (false)
#else
#define DEBUG_PRINT(...) \
    do                   \
    {                    \
    } while (false)
#endif

int shapeSize(int *shape, int ndim)
{
    int prod = 1;
    for (int i = 0; i < ndim; i++)
    {
        prod *= shape[i];
    }
    return prod;
}

int validateAxis(int axis, int ndim)
{
    if (axis >= ndim)
    {
        return -1;
    }
    if (axis < 0)
    {
        axis += ndim;
    }
    if (axis < 0)
    {
        return -2;
    }
    return axis;
}

void printIntArray(int *row, int length)
{
    printf("[");
    if (length == 0)
    {
        printf("]");
        return;
    }

    for (int i = 0; i < length - 1; i++)
    {
        printf("%d, ", row[i]);
    }
    printf("%d]", row[length - 1]);
}

void printFloatArray(float *row, int length)
{
    printf("[");
    if (length == 0)
    {
        printf("]");
        return;
    }

    for (int i = 0; i < length - 1; i++)
    {
        printf("%.2f, ", row[i]);
    }
    printf("%.2f]", row[length - 1]);
}

void NDArray_decRefCount(struct NDArray *array)
{
    --*array->refCount;
    if (*array->refCount == 0)
    {
        free(array->data);
        free(array->refCount);
    }
}

struct NDArray *NDArray_create(int *shape, int ndim, NDARRAY_TYPE *data)
{
    int dataCount = shapeSize(shape, ndim);

    struct NDArray *output = (struct NDArray *)malloc(sizeof(struct NDArray));
    output->data = data;
    output->refCount = (int *)malloc(sizeof(int));
    *output->refCount = 1;

    output->steps = (int *)malloc(sizeof(int) * ndim);
    output->shape = (int *)malloc(sizeof(int) * ndim);
    memcpy(output->shape, shape, sizeof(int) * ndim);
    int prod = 1;
    for (int i = ndim - 1; i >= 0; i--)
    {
        output->steps[i] = prod;
        prod *= shape[i];
    }
    output->steps[ndim - 1] = 1;

    output->ndim = ndim;
    output->dataCount = dataCount;

    return output;
}

struct NDArray *NDArray_zeros(int *shape, int ndim)
{
    int dataCount = shapeSize(shape, ndim);
    NDARRAY_TYPE *data = (NDARRAY_TYPE *)calloc(dataCount, sizeof(NDARRAY_TYPE));

    return NDArray_create(shape, ndim, data);
}

struct NDArray *NDArray_ones(int *shape, int ndim)
{
    int dataCount = shapeSize(shape, ndim);
    NDARRAY_TYPE *data = (NDARRAY_TYPE *)malloc(dataCount * sizeof(NDARRAY_TYPE));
    for (int i = 0; i < dataCount; i++)
    {
        data[i] = 1;
    }

    return NDArray_create(shape, ndim, data);
}

struct NDArray *NDArray_single(NDARRAY_TYPE value, int ndim)
{
    NDARRAY_TYPE *data = (NDARRAY_TYPE *)malloc(sizeof(NDARRAY_TYPE));
    *data = value;

    int shape[ndim];
    for (int i = 0; i < ndim; i++)
    {
        shape[i] = 1;
    }

    return NDArray_create(shape, ndim, data);
}

struct NDArray *NDArray_eye(int size)
{
    int shape[] = {size, size};
    struct NDArray *output = NDArray_zeros(shape, 2);

    for (int i = 0; i < size; i++)
    {
        output->data[i * size + i] = 1;
    }

    return output;
}

// Assumes index is valid. Sets all to zero if array is finished.
// Otherwise, return step
int NDArray_incIndex(struct NDArray *array, int *index)
{
    int step = 0;
    for (int i = array->ndim - 1; i >= 0; i--)
    {
        index[i]++;
        if (index[i] == array->shape[i])
        {
            index[i] = 0;
            step -= (array->shape[i] - 1) * array->steps[i];
        }
        else
        {
            return step + array->steps[i];
        }
    }
    // This is really just a flag. I think the step to loop should really be this - 1
    return -array->dataCount;
}

int NDArray_reshape(struct NDArray *array, int *newShape, int newNDim)
{
    // Copy the shape, to avoid modifying the original
    int tempNewShape[newNDim];
    memcpy(tempNewShape, newShape, newNDim * sizeof(int));
    newShape = tempNewShape;

    // Calculate the value of an undefined index
    int undefIndex = -1;
    int prod = 1;
    bool identical = array->ndim == newNDim;
    for (int i = 0; i < newNDim; i++)
    {
        if (newShape[i] == -1)
        {
            if (undefIndex != -1)
            {
                return 2;
            }
            else
            {
                undefIndex = i;
            }
        }
        else
        {
            // Check if the
            if (identical)
            {
                identical = array->shape[i] == newShape[i];
            }
            prod *= newShape[i];
        }
    }

    if (undefIndex != -1)
    {
        newShape[undefIndex] = array->dataCount / prod;
    }

    // Check if the new shape is actually valid
    if (shapeSize(newShape, newNDim) != array->dataCount)
    {
        return 1;
    }

    // Only shortcut out after checking that the new shape is valid. It probably has to be anyway
    if (identical)
    {
        return 0;
    }

    int indOld = array->ndim - 1;
    int indNew = newNDim - 1;
    int newSteps[newNDim];
    int oldShape[array->ndim];
    memcpy(oldShape, array->shape, array->ndim * sizeof(int));
    int oldSteps[array->ndim];
    memcpy(oldSteps, array->steps, array->ndim * sizeof(int));
    if (newShape[indNew] == 1)
    {
        newSteps[indNew] = 1;
        indNew--;
    }
    while (indNew >= 0)
    {
        if (oldShape[indOld] % newShape[indNew] == 0)
        {
            newSteps[indNew] = oldSteps[indOld];
            oldSteps[indOld] *= newShape[indNew];
            oldShape[indOld] /= newShape[indNew];
            indNew--;
        }
        else if (oldShape[indOld] == 1 || oldSteps[indOld - 1] == oldShape[indOld] * oldSteps[indOld])
        {
            oldShape[indOld - 1] *= oldShape[indOld];
            indOld--;
        }
        else
        {
            DEBUG_PRINT("Making contiguous!\n");
            NDArray_makeContiguous(array);
            newSteps[newNDim - 1] = 1;
            for (int i = newNDim - 1; i > 0; i--)
            {
                newSteps[i - 1] = newShape[i] * newSteps[i];
            }
            break;
        }
    }

    // Set new shape and steps
    array->shape = (int *)realloc(array->shape, sizeof(int) * newNDim);
    memcpy(array->shape, newShape, sizeof(int) * newNDim);

    array->steps = (int *)realloc(array->steps, sizeof(int) * newNDim);
    memcpy(array->steps, newSteps, sizeof(int) * newNDim);

    array->ndim = newNDim;
    return 0;
}

int NDArray_expandDims(struct NDArray *array, int axis)
{
    axis = validateAxis(axis, array->ndim + 1);
    if (axis < 0)
    {
        return 1;
    }

    int newShape[array->ndim + 1];
    for (int i = 0; i < array->ndim; i++)
    {
        newShape[i + (i >= axis)] = array->shape[i];
    }
    newShape[axis] = 1;
    int errorCode = NDArray_reshape(array, newShape, array->ndim + 1);
    if (errorCode > 0)
    {
        // Offset errorCode past internal errors
        return errorCode + 1;
    }
    return 0;
}

int NDArray_squeeze(struct NDArray *array, int axis)
{
    axis = validateAxis(axis, array->ndim);
    if (axis < 0)
    {
        return 1;
    }
    if (array->shape[axis] != 1)
    {
        return 2;
    }

    int newShape[array->ndim - 1];
    for (int i = 0; i < array->ndim - 1; i++)
    {
        newShape[i] = array->shape[i + (i >= axis)];
    }
    int errorCode = NDArray_reshape(array, newShape, array->ndim - 1);
    if (errorCode > 0)
    {
        // Offset errorCode past internal errors
        return errorCode + 2;
    }
    return 0;
}

int NDArray_transpose(struct NDArray *array, int *newOrder)
{
    int *tempSteps = (int *)malloc(sizeof(int) * array->ndim);
    int *tempShape = (int *)malloc(sizeof(int) * array->ndim);

    for (int i = 0; i < array->ndim; i++)
    {
        tempSteps[i] = array->steps[newOrder[i]];
        tempShape[i] = array->shape[newOrder[i]];
    }
    free(array->steps);
    free(array->shape);
    array->steps = tempSteps;
    array->shape = tempShape;
    return 0;
}

int NDArray_swapAxes(struct NDArray *array, int axis1, int axis2)
{
    axis1 = validateAxis(axis1, array->ndim);
    axis2 = validateAxis(axis2, array->ndim);

    if (axis1 < 0 || axis2 < 0)
    {
        return 1;
    }

    int newOrder[array->ndim];
    for (int i = 0; i < array->ndim; i++)
    {
        newOrder[i] = i;
    }
    newOrder[axis1] = axis2;
    newOrder[axis2] = axis1;

    int errorCode = NDArray_transpose(array, newOrder);
    if (errorCode != 0)
    {
        return errorCode + 1;
    }
    return 0;
}

NDARRAY_TYPE *NDArray_getPointer(struct NDArray *array, int *index)
{
    NDARRAY_TYPE *addr = array->data;
    for (int i = 0; i < array->ndim; i++)
    {
        addr += index[i] * array->steps[i];
    }
    return addr;
}

int compare(const void *a, const void *b)
{
    return (*(int *)b - *(int *)a);
}

int NDArray_makeContiguous(struct NDArray *array)
{
    NDARRAY_TYPE *newData = (NDARRAY_TYPE *)malloc(array->dataCount * sizeof(NDARRAY_TYPE));
    int index[array->ndim];
    memset(index, 0, array->ndim * sizeof(int));

    int nextStep;
    NDARRAY_TYPE *outPointer = newData;
    do
    {
        *outPointer = NDArray_get(array, index);
        outPointer++;
    } while (NDArray_incIndex(array, index) != -array->dataCount);

    NDArray_decRefCount(array);
    array->data = newData;
    array->refCount = (int *)malloc(sizeof(int));
    *array->refCount = 1;

    qsort(array->steps, array->ndim, sizeof(int), compare);
    return 0;
}

// Assumes pointer is in array. Currently, this won't work with steps of 0
void NDArray_getIndex(struct NDArray *array, NDARRAY_TYPE *pointer, int *index)
{
    int offset = pointer - array->data;
    int prevStep = array->dataCount;
    for (int i = 0; i < array->ndim; i++)
    {
        // Get the largest step that is smaller than the previous step
        int step = 0;
        int stepIndex = 0;
        for (int j = 0; j < array->ndim; j++)
        {
            if (array->steps[j] < prevStep && array->steps[j] > step)
            {
                step = array->steps[j];
                stepIndex = j;
            }
        }
        prevStep = step;

        index[stepIndex] = offset / step;
        offset = offset % step;
    }
}

// I feel there is a better way of doing this...
// NDARRAY_TYPE *NDArray_nextPointer(struct NDArray *array, NDARRAY_TYPE *pointer)
// {
//     if (pointer - array->data == array->dataCount - 1)
//     {
//         return 0;
//     }

//     int index[array->ndim];
//     NDArray_getIndex(array, pointer, index);

//     int nextStep = NDArray_incIndex(array, index);

//     if (nextStep == 0)
//     {
//         printf("I didn't think this should ever happen");
//         return 0;
//     }
//     else
//     {
//         return pointer + nextStep;
//     }
// }

NDARRAY_TYPE NDArray_get(struct NDArray *array, int *index)
{
    return *NDArray_getPointer(array, index);
}

void NDArray_set(struct NDArray *array, int *index, NDARRAY_TYPE value)
{
    *NDArray_getPointer(array, index) = value;
}

void NDArray_free(struct NDArray *array)
{
    if (array != 0)
    {
        free(array->steps);
        free(array->shape);
        NDArray_decRefCount(array);
        free(array);
    }
}

void printSubArray(struct NDArray *array, int indent, int *index)
{
    for (int i = 0; i < indent; i++)
    {
        printf("   ");
    }
    printf("[");

    // If this is the deepest layer in the array
    if (indent == array->ndim - 1)
    {
        for (int i = 0; i < array->shape[indent] - 1; i++)
        {
            printf(NDARRAY_TYPE_FORMAT ", ", NDArray_get(array, index));
            // printf("%d, ", (int)pointer / 4);
            NDArray_incIndex(array, index);
        }
        printf(NDARRAY_TYPE_FORMAT, NDArray_get(array, index));
        // printf("%d", (int)pointer / 4);
        NDArray_incIndex(array, index);
    }
    else
    {
        printf("\n");
        for (int i = 0; i < array->shape[indent]; i++)
        {
            printSubArray(array, indent + 1, index);
        }

        for (int i = 0; i < indent; i++)
        {
            printf("   ");
        }
    }

    printf("]\n");
}

void NDArray_print(struct NDArray *array)
{
    int ndim = array->ndim;
    printf("Shape: ");
    printIntArray(array->shape, ndim);
    printf("\nSteps: ");
    printIntArray(array->steps, ndim);
    printf("\n");

    int index[array->ndim];
    memset(index, 0, array->ndim * sizeof(int));
    printSubArray(array, 0, index);
}

// Shape is assumed to have the size of array->ndim
struct NDArray *NDArray_broadcastTo(struct NDArray *array, int *shape)
{
    // Check for incompatible shapes
    for (int i = 0; i < array->ndim; i++)
    {
        if (array->shape[i] != shape[i] && array->shape[i] != 1)
        {
            DEBUG_PRINT("Incompatible shapes\n");
            return 0;
        }
    }

    struct NDArray *result = (struct NDArray *)malloc(sizeof(struct NDArray));
    result->data = array->data;
    result->refCount = array->refCount;
    ++*result->refCount;
    result->ndim = array->ndim;

    result->steps = (int *)malloc(sizeof(int) * array->ndim);
    result->shape = (int *)malloc(sizeof(int) * array->ndim);
    memcpy(result->shape, shape, sizeof(int) * array->ndim);
    memcpy(result->steps, array->steps, sizeof(int) * array->ndim);

    // Convert all steps that are different to 0
    for (int i = 0; i < result->ndim; i++)
    {
        if (array->shape[i] != result->shape[i])
        {
            result->steps[i] = 0;
        }
    }

    result->ndim = array->ndim;
    result->dataCount = array->dataCount;
    return result;
}

struct NDArrayPair NDArray_broadcast(struct NDArray *a, struct NDArray *b)
{
    struct NDArrayPair arrayPair = {0, 0};
    if (a->ndim != b->ndim)
    {
        return arrayPair;
    }

    int newShape[a->ndim];
    for (int i = 0; i < a->ndim; i++)
    {
        if (a->shape[i] == b->shape[i] || b->shape[i] == 1)
        {
            newShape[i] = a->shape[i];
        }
        else if (a->shape[i] == 1)
        {
            newShape[i] = b->shape[i];
        }
        else
        {
            return arrayPair;
        }
    }

    arrayPair.a = NDArray_broadcastTo(a, newShape);
    arrayPair.b = NDArray_broadcastTo(b, newShape);

    return arrayPair;
}

struct NDArray *NDArray_sum(struct NDArray *array, int axis)
{
    axis = validateAxis(axis, array->ndim);

    if (axis < 0 || axis >= array->ndim)
    {
        return 0;
    }
    if (array->ndim < 2)
    {
        return 0;
    }

    // The new shape is a copy of the old, except the sum axis
    int shape[array->ndim];
    memcpy(shape, array->shape, array->ndim * sizeof(int));
    shape[axis] = 1;

    struct NDArray *result = NDArray_zeros(shape, array->ndim);

    struct NDArray *bResult = NDArray_broadcastTo(result, array->shape);

    // Step through all elements in the input array and add them to the
    // corresponding output element (which is repeated along the axis).
    // Part of this is similar to nextPointer, but it's more efficient
    // to implement it here
    int index[array->ndim];
    memset(index, 0, array->ndim * sizeof(int));
    int nextStep;
    NDARRAY_TYPE *pointer = array->data;
    do
    {
        *NDArray_getPointer(bResult, index) += NDArray_get(array, index);

        nextStep = NDArray_incIndex(array, index);
    } while (nextStep != -array->dataCount);
    NDArray_free(bResult);

    // The new shape is a copy of the old, excluding the sum axis
    int outShape[array->ndim - 1];
    for (int i = 0; i < array->ndim - 1; i++)
    {
        outShape[i] = array->shape[i + (i >= axis)];
    }

    NDArray_reshape(result, outShape, array->ndim - 1);
    return result;
}

struct NDArray *NDArray_multiply(struct NDArray *a, struct NDArray *b)
{
    struct NDArrayPair pair = NDArray_broadcast(a, b);
    if (pair.a == 0)
    {
        return 0;
    }

    int index[pair.a->ndim];
    memset(index, 0, pair.a->ndim * sizeof(int));

    struct NDArray *result = NDArray_zeros(pair.a->shape, pair.a->ndim);

    NDARRAY_TYPE *pointer = result->data;
    do
    {
        *pointer = NDArray_get(pair.a, index) * NDArray_get(pair.b, index);
        pointer++;
    } while (NDArray_incIndex(pair.a, index) != -pair.a->dataCount);

    NDArray_free(pair.a);
    NDArray_free(pair.b);
    return result;
}

struct NDArray *NDArray_add(struct NDArray *a, struct NDArray *b)
{
    struct NDArrayPair pair = NDArray_broadcast(a, b);
    if (pair.a == 0)
    {
        return 0;
    }

    int index[pair.a->ndim];
    memset(index, 0, pair.a->ndim * sizeof(int));

    struct NDArray *result = NDArray_zeros(pair.a->shape, pair.a->ndim);

    NDARRAY_TYPE *pointer = result->data;
    do
    {
        *pointer = NDArray_get(pair.a, index) + NDArray_get(pair.b, index);
        pointer++;
    } while (NDArray_incIndex(pair.a, index) != -pair.a->dataCount);

    NDArray_free(pair.a);
    NDArray_free(pair.b);
    return result;
}

struct NDArray *NDArray_copy(struct NDArray *array)
{
    struct NDArray *output = (struct NDArray *)malloc(sizeof(struct NDArray));
    output->data = array->data;
    output->refCount = array->refCount;
    ++*output->refCount;
    output->ndim = array->ndim;
    output->dataCount = array->dataCount;

    output->steps = (int *)malloc(sizeof(int) * array->ndim);
    output->shape = (int *)malloc(sizeof(int) * array->ndim);
    memcpy(output->shape, array->shape, sizeof(int) * array->ndim);
    memcpy(output->steps, array->steps, sizeof(int) * array->ndim);

    return output;
}

struct NDArray *NDArray_clone(struct NDArray *array)
{
    struct NDArray *output = (struct NDArray *)malloc(sizeof(struct NDArray));

    output->ndim = array->ndim;
    output->dataCount = array->dataCount;

    output->data = (NDARRAY_TYPE *)malloc(sizeof(NDARRAY_TYPE) * array->dataCount);
    memcpy(output->data, array->data, sizeof(NDARRAY_TYPE) * array->dataCount);

    output->refCount = (int *)malloc(sizeof(int));
    *output->refCount = 1;

    output->steps = (int *)malloc(sizeof(int) * array->ndim);
    output->shape = (int *)malloc(sizeof(int) * array->ndim);
    memcpy(output->shape, array->shape, sizeof(int) * array->ndim);
    memcpy(output->steps, array->steps, sizeof(int) * array->ndim);

    return output;
}

struct NDArray *NDArray_matmul(struct NDArray *a, struct NDArray *b)
{
    // TODO: Implement this
    if (a->ndim != b->ndim)
    {
        return 0;
    }
    int ndim = a->ndim;
    if (ndim < 2)
    {
        return 0;
    }

    a = NDArray_copy(a);
    b = NDArray_copy(b);

    int errorCode = 0;
    errorCode |= NDArray_expandDims(a, -3);
    errorCode |= NDArray_expandDims(b, -3);

    struct NDArray *output = 0;

    if (errorCode == 0)
    {
        errorCode |= NDArray_swapAxes(a, -1, -3);
        errorCode |= NDArray_swapAxes(b, -2, -3);
    }
    if (errorCode == 0)
    {
        struct NDArray *prod = NDArray_multiply(a, b);
        if (prod == 0)
        {
            errorCode |= 1;
        }

        if (errorCode == 0)
        {
            output = NDArray_sum(prod, -3);
            if (output == 0)
            {
                errorCode |= 1;
            }
        }
        NDArray_free(prod);
    }

    NDArray_free(a);
    NDArray_free(b);
    return output;
}

struct NDArray *NDArray_inv(struct NDArray *array)
{
    // This is a naive implementation of Gaussian elimination. Better performance may be
    // obtained from https://sites.engineering.ucsb.edu/~hpscicom/projects/gauss/introge.pdf
    // This doesn't actually check if a matrix is invertable
    if (array->ndim < 2)
    {
        return 0;
    }
    if (array->shape[array->ndim - 1] != array->shape[array->ndim - 2])
    {
        return 0;
    }

    struct NDArray *input = NDArray_clone(array);

    int shape[3];
    shape[0] = -1;
    shape[1] = input->shape[input->ndim - 2];
    shape[2] = input->shape[input->ndim - 1];
    NDArray_reshape(input, shape, 3);

    int n = input->shape[input->ndim - 1];

    struct NDArray *output = NDArray_zeros(input->shape, input->ndim);

    int index[3];
    NDARRAY_TYPE x, temp1, temp2;
    for (int b = 0; b < input->shape[0]; b++)
    {
        index[0] = b;
        // Convert zeros to identity
        for (int i = 0; i < n; i++)
        {
            index[1] = i;
            index[2] = i;
            NDArray_set(output, index, 1);
        }

        for (int i = 0; i < n; i++)
        {
            index[1] = i;
            index[2] = i;
            if (NDArray_get(input, index) == 0)
            {
                for (int ii = 0; ii < n; ii++)
                {
                    index[1] = i;
                    index[2] = ii;
                    temp1 = NDArray_get(input, index);
                    index[1] = ii;
                    index[2] = i;
                    temp2 = NDArray_get(input, index);
                    if (ii != i && temp1 != 0 && temp2 != 0)
                    {
                        for (int j = 0; j < n; j++)
                        {
                            index[2] = j;

                            index[1] = i;
                            temp1 = NDArray_get(input, index);
                            index[1] = ii;
                            temp2 = NDArray_get(input, index);
                            NDArray_set(input, index, temp1);
                            index[1] = i;
                            NDArray_set(input, index, temp2);

                            temp1 = NDArray_get(output, index);
                            index[1] = ii;
                            temp2 = NDArray_get(output, index);
                            NDArray_set(output, index, temp1);
                            index[1] = i;
                            NDArray_set(output, index, temp2);
                        }
                        break;
                    }
                }
            }
        }

        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < i; k++)
            {
                index[1] = i;
                index[2] = k;
                x = NDArray_get(input, index);
                for (int j = 0; j < n; j++)
                {
                    index[1] = k;
                    index[2] = j;
                    temp1 = NDArray_get(input, index);
                    temp2 = NDArray_get(output, index);
                    index[1] = i;
                    *NDArray_getPointer(input, index) -= x * temp1;
                    *NDArray_getPointer(output, index) -= x * temp2;
                }
            }

            index[1] = i;
            index[2] = i;
            x = 1 / NDArray_get(input, index);
            for (int j = 0; j < n; j++)
            {
                index[2] = j;
                *NDArray_getPointer(input, index) *= x;
                *NDArray_getPointer(output, index) *= x;
            }
        }

        for (int i = n - 1; i > 0; i--)
        {
            for (int ii = i - 1; ii >= 0; ii--)
            {
                index[1] = ii;
                index[2] = i;
                x = NDArray_get(input, index);
                for (int j = 0; j < n; j++)
                {
                    index[1] = i;
                    index[2] = j;
                    temp1 = NDArray_get(input, index);
                    temp2 = NDArray_get(output, index);
                    index[1] = ii;
                    *NDArray_getPointer(input, index) -= x * temp1;
                    *NDArray_getPointer(output, index) -= x * temp2;
                }
            }
        }
    }

    NDArray_reshape(output, array->shape, array->ndim);
    NDArray_free(input);
    return output;
}
#ifndef NDARRAY_DEFINED
#define NDARRAY_DEFINED

#ifndef NDARRAY_TYPE
#define NDARRAY_TYPE float
#define NDARRAY_TYPE_FORMAT "%4.4f"
#endif

#ifdef __cplusplus
 extern "C" {
#endif

struct NDArray
{
    int *steps;
    int *shape;
    int ndim;
    int dataCount;
    NDARRAY_TYPE *data;
    int *refCount;
};

struct NDArrayPair
{
    struct NDArray *a;
    struct NDArray *b;
};

struct NDArray *NDArray_eye(int size);

struct NDArray *NDArray_zeros(int *shape, int ndim);

struct NDArray *NDArray_ones(int *shape, int ndim);

struct NDArray *NDArray_single(NDARRAY_TYPE value, int ndim);

int NDArray_reshape(struct NDArray *array, int *newShape, int newNDim);

void NDArray_free(struct NDArray *array);

int NDArray_transpose(struct NDArray *array, int *newOrder);

int NDArray_swapAxes(struct NDArray *array, int axis1, int axis2);

NDARRAY_TYPE NDArray_get(struct NDArray *array, int *index);

void NDArray_set(struct NDArray *array, int *index, NDARRAY_TYPE value);

// NDARRAY_TYPE *NDArray_nextPointer(struct NDArray *array, NDARRAY_TYPE *pointer);

int NDArray_makeContiguous(struct NDArray *array);

void NDArray_print(struct NDArray *array);

struct NDArray *NDArray_sum(struct NDArray *array, int axis);

void printIntArray(int *row, int length);

void printFloatArray(float *row, int length);

struct NDArray *NDArray_broadcastTo(struct NDArray *array, int *shape);

struct NDArrayPair NDArray_broadcast(struct NDArray *a, struct NDArray *b);

int NDArray_expandDims(struct NDArray *array, int axis);

int NDArray_squeeze(struct NDArray *array, int axis);

struct NDArray *NDArray_multiply(struct NDArray *a, struct NDArray *b);

struct NDArray *NDArray_add(struct NDArray *a, struct NDArray *b);

struct NDArray *NDArray_matmul(struct NDArray *a, struct NDArray *b);

struct NDArray *NDArray_inv(struct NDArray *array);

struct NDArray *NDArray_copy(struct NDArray *array);

struct NDArray *NDArray_clone(struct NDArray *array);

#endif

#ifdef __cplusplus
}
#endif
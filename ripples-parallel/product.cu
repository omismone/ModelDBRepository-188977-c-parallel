#include "omislib.cuh"

void productMatMat(struct matrix* c, struct matrix* a, struct matrix* b)
{
    double* dev_a = 0;
    double* dev_b = 0;
    double* dev_c = 0;

    cublasHandle_t handle;

    cudaSetDevice(0);

    cublasCreate(&handle);

    cudaMalloc((void**)&dev_a, a->size[0] * a->size[1] * sizeof(double));
    cudaMalloc((void**)&dev_b, b->size[0] * b->size[1] * sizeof(double));
    cudaMalloc((void**)&dev_c, c->size[0] * c->size[1] * sizeof(double));

    cudaMemcpy(dev_a, a->val, a->size[0] * a->size[1] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b->val, b->size[0] * b->size[1] * sizeof(double), cudaMemcpyHostToDevice);

    const double alpha = 1.0;
    const double beta = 0.0;
    int m = b->size[1];
    int n = a->size[0];
    int k = b->size[0];
    cublasStatus_t stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, dev_b, n, dev_a, k, &beta, dev_c, n);

    cudaMemcpy(c->val, dev_c, c->size[0] * c->size[1] * sizeof(double), cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}
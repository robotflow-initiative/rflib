#include "pytorch_cpp_helper.hpp"
#ifdef RFLIB_WITH_CUDA
#include <THC/THC.h>
void knn_device(float* ref_dev, int ref_nb, float* query_dev, int query_nb,
    int dim, int k, float* dist_dev, long* ind_dev, cudaStream_t stream);
#endif



void knn_cpu(float* ref_dev, int ref_width, float* query_dev, int query_width,
    int height, int k, float* dist_dev, long* ind_dev, long* ind_buf)
{
    // Compute all the distances
    for (int query_idx = 0; query_idx < query_width; query_idx++)
    {
        for (int ref_idx = 0; ref_idx < ref_width; ref_idx++)
        {
            dist_dev[query_idx * ref_width + ref_idx] = 0;
            for (int hi = 0; hi < height; hi++)
                dist_dev[query_idx * ref_width + ref_idx] += (ref_dev[hi * ref_width + ref_idx] - query_dev[hi * query_width + query_idx]) * (ref_dev[hi * ref_width + ref_idx] - query_dev[hi * query_width + query_idx]);
        }
    }

    float temp_value;
    long temp_idx;
    // sort the distance and get the index
    for (int query_idx = 0; query_idx < query_width; query_idx++)
    {
        for (int i = 0; i < ref_width; i++)
        {
            ind_buf[i] = i + 1;
        }
        for (int i = 0; i < ref_width; i++)
            for (int j = 0; j < ref_width - i - 1; j++)
            {
                if (dist_dev[query_idx * ref_width + j] > dist_dev[query_idx * ref_width + j + 1])
                {
                    temp_value = dist_dev[query_idx * ref_width + j];
                    dist_dev[query_idx * ref_width + j] = dist_dev[query_idx * ref_width + j + 1];
                    dist_dev[query_idx * ref_width + j + 1] = temp_value;
                    temp_idx = ind_buf[j];
                    ind_buf[j] = ind_buf[j + 1];
                    ind_buf[j + 1] = temp_idx;
                }
            }
        for (int i = 0; i < k; i++)
            ind_dev[query_idx + i * query_width] = ind_buf[i];
    }
}


void knn(at::Tensor& ref, at::Tensor& query, at::Tensor& idx)
{

    // TODO check dimensions
    long batch, ref_nb, query_nb, dim, k;
    batch = ref.size(0);
    dim = ref.size(1);
    k = idx.size(1);
    ref_nb = ref.size(2);
    query_nb = query.size(2);

    float* ref_dev = ref.data_ptr<float>();
    float* query_dev = query.data_ptr<float>();
    long* idx_dev = idx.data_ptr<long>();

    if (ref.is_cuda()) {
#ifdef RFLIB_WITH_CUDA
        extern THCState* state;
        // TODO raise error if not compiled with CUDA
        float* dist_dev = (float*)THCudaMalloc(state, ref_nb * query_nb * sizeof(float));

        for (int b = 0; b < batch; b++)
        {
            knn_device(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
                dist_dev, idx_dev + b * k * query_nb, c10::cuda::getCurrentCUDAStream());
        }
        THCudaFree(state, dist_dev);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("error in knn: %s\n", cudaGetErrorString(err));
            THError("aborting");
        }
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    float* dist_dev = (float*)malloc(ref_nb * query_nb * sizeof(float));
    long* ind_buf = (long*)malloc(ref_nb * sizeof(long));
    for (int b = 0; b < batch; b++) {
        knn_cpu(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
            dist_dev, idx_dev + b * k * query_nb, ind_buf);
    }
    free(dist_dev);
    free(ind_buf);
}


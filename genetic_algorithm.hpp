#pragma once

#include <vector>
#include <stddef.h>
#include <cuda_runtime.h>
#include <nvfunctional>
#include <cstdio>
#include "genetic_algorithm_cuda.cuh"

/* Check errors on CUDA runtime functions */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
printf("Error at %s:%d\n",__FILE__,__LINE__);\
return EXIT_FAILURE;}} while(0)
// See more at: http://docs.nvidia.com/cuda/curand/host-api-overview.html#generation-functions

/* Check errors on CUDA kernel calls */
void checkCUDAKernelError()
{
    cudaError_t err = cudaGetLastError();
    if  (cudaSuccess != err){
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    }
    
}

// TODO: Potentially make generic using template parameters. This might hurt performance though
//       since we'd have to introduce another lambda.
namespace genetic_algorithm {
    template <typename Spawn, typename Evaluate, typename Mutate, typename Cross>
    static void simulate(float *results, specification_t specification, Spawn spawn, Evaluate evaluate, Mutate mutate, Cross cross) {
        unsigned blocks = specification.kingdom_count * specification.island_count_per_kingdom;
        unsigned threadsPerBlock = specification.island_population;
        
        genome_t * dev_inboxes;
        gpuErrchk(cudaMalloc((void **)&dev_inboxes, blocks * sizeof(genome_t)));
        
        cudaCallSimulation(blocks, threadsPerBlock, dev_inboxes, specification, spawn, evaluate, mutate, cross);
        checkCUDAKernelError();
        
        gpuErrchk(cudaMemcpy(
                             results,
                             dev_inboxes,
                             specification.kingdom_count * sizeof(genome_t),
                             cudaMemcpyDeviceToHost
                             ));
        cudaFree(dev_inboxes);
    }
};


#include "genetic_algorithm.hpp"
#include "genetic_algorithm_cuda.cuh"
#include <stdlib.h>
#include <string.h>
#include <cstdio>

// Runs code on each block sequentially, declaring a few local variables in the process.
// Requires that `blocks` and `threadsPerBlock` be defined in the local scope.
#define foreach_block(braced_code) ({\
unsigned blockDim_x = threadsPerBlock;\
(void)blockDim_x;\
for (unsigned blockIdx_x = 0; blockIdx_x < blocks; blockIdx_x++) {\
/* For CPU version, only run if we shouldn't have exited the loop */\
if (finished_blocks[blockIdx_x]) continue;\
/* Compute the location of *our* shared memory */\
thread_data *shared_memory = (thread_data *)((char *)__all_block_shared_memory + blockIdx_x * bytesPerBlock);\
island_index index = (island_index){\
.kingdom_index = blockIdx_x / specification.island_count_per_kingdom,\
.cross_section_index = blockIdx_x % specification.island_count_per_kingdom\
};\
(void)index;\
(braced_code);\
}\
})\

// Runs code on each thread sequentially, declaring a few local variables in the process.
// Should only be used inside `foreach_block`.
#define foreach_thread(braced_code) ({\
for (unsigned threadIdx_x = 0; threadIdx_x < blockDim_x; threadIdx_x++) {\
thread_data *thread_memory = shared_memory + threadIdx_x;\
(braced_code);\
}\
})

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

namespace genetic_algorithm {
    void simulation::run(specification_t specification, size_t max_iterations, float * const results) {
        unsigned blocks = specification.kingdom_count * specification.island_count_per_kingdom;
        unsigned threadsPerBlock = specification.island_population;
        
        genome_t * dev_inboxes;
        gpuErrchk(cudaMalloc((void **)&dev_inboxes, blocks * sizeof(genome_t)));
        
        cudaCallSimulation(blocks, threadsPerBlock, dev_inboxes, specification, operations, max_iterations);
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


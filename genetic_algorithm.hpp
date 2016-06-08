#pragma once

#include <vector>
#include <stddef.h>
#include <cuda_runtime.h>
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

namespace genetic_algorithm {
    /* Since template arguments must be used for the specification of the algorithm,
       we've included documentation of what the type signature ought to be for the
       code to correclty compile.
     
     // Return a random genome.
     nvstd::function<genome_t(island_index index, curandState_t *rand_state)> spawn;
     
     // Evaluate the `test_genome` returning its fitness.
     // Note that `competitor_genomes` is an array of the best genomes in each kingdom,
     // ordered by kingdom, fomr `test_genomes` cross section of the world.
     nvstd::function<fitness_t(island_index index, genome_t test_genome, genome_t *competitor_genomes, curandState_t *rand_state)> evaluate;
     
     // DECIDE: Should we allow the caller to determine whether or not its mutated like this?
     // Mutate a genome or leave it alone.
     nvstd::function<void(island_index index, size_t genome_index, genome_t *genome, curandState_t *rand_state)> mutate;
     
     // DECIDE: Any reason to share the genome indices here? It seems inconsistent to not.
     // Cross two genomes.
     nvstd::function<genome_t(island_index index, genome_t genome_x, genome_t genome_y, curandState_t *rand_state)> cross;
     */
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

#pragma once

#include <curand.h>
#include <stdio.h>
#include <curand_kernel.h>

// Helpers for accessing a specific inbox by index
#define mailboxes(cross_section) (inboxes + (cross_section) * specification.kingdom_count)
#define inbox(index) (mailboxes((index).cross_section_index) + (index).kingdom_index)

namespace genetic_algorithm {
    // Floats are small enough such that they can be atomically moved between blocks without
    // corruption. If we change this to a different size later on, we'll have to introduce
    // synchronization.
    typedef float fitness_t;
    typedef float genome_t;
    
    // Stored in shared memory
    struct thread_data {
        genome_t genome;
        fitness_t fitness;
        curandState rand_state;
    };
    
    // Stored on stack
    struct island_index {
        size_t kingdom_index;
        size_t cross_section_index;
    };
    
    struct specification_t {
        size_t kingdom_count;
        size_t island_count_per_kingdom;
        size_t island_population;
    };
    
    template <typename Spawn, typename Evaluate, typename Mutate, typename Cross>
    __global__
    void cudaSimulationKernel(genome_t *inboxes, specification_t specification, size_t max_iterations, Spawn spawn, Evaluate evaluate, Mutate mutate, Cross cross) {
        extern __shared__ thread_data shared_memory[];
        thread_data *thread_memory = shared_memory + threadIdx.x;
        
        island_index index;
        index.kingdom_index = blockIdx.x / specification.island_count_per_kingdom;
        index.cross_section_index = blockIdx.x % specification.island_count_per_kingdom;
        //    island_index index = (island_index){
        //        .kingdom_index = blockIdx.x / specification.island_count_per_kingdom,
        //        .cross_section_index = blockIdx.x % specification.island_count_per_kingdom
        //    };
        
        // Seed the random number generator
        // Note we can simply see with the thread index since we don't care to have random
        // behavior between multiple invocations, only between threads
        curand_init(blockIdx.x * blockDim.x + threadIdx.x, 1, 0, &thread_memory->rand_state);
        
        // Initialize individual
        //    auto x = operations.spawn;
            printf("%f\n", spawn(index, &thread_memory->rand_state));
        //    thread_memory->genome = operations.spawn(index, &thread_memory->rand_state);
        
        /*
        for (size_t i = 0; i < max_iterations; i++) {
            // Evaluate individual
            thread_memory->fitness = operations.evaluate(index, thread_memory->genome, mailboxes(index.cross_section_index), &thread_memory->rand_state);
            
            // Perform reduction
            for (size_t bound = blockDim.x >> 1; bound > 0; bound >>= 1) {
                thread_data *other_memory = thread_memory + bound;
                syncthreads();
                
                // TODO: Allow selection between minimum and maximum.
                // Copy the maximum value of thread and other to the opposite
                if (thread_memory->fitness > other_memory->fitness) {
                    // Note that this case must be first since comparison with NaN
                    // is always false.
                    thread_memory->genome = other_memory->genome;
                    thread_memory->fitness = other_memory->fitness;
                } else {
                    // Idea: If we end up with too many of the largest value,
                    //       we could probabilistically skip this branch.
                    other_memory->genome = thread_memory->genome;
                    other_memory->fitness = thread_memory->fitness;
                }
            }
            
            // Perform migration of elite individual
            if (threadIdx.x == 0) {
                // Outward
                {
                    // Send updated elite genome to next island in kingdom
                    //                island_index next_index = ((island_index){
                    //                    .kingdom_index = index.kingdom_index,
                    //                    .cross_section_index = (index.cross_section_index + 1) % specification.island_count_per_kingdom
                    //                });
                    // Note that this thread's genome in shared memory is the best since we reduced left
                    //                *inbox(next_index) = thread_memory->genome;
                }
                
                // Inward
                {
                    // Migrate elite individual from island into the population
                    // Since thread 2 has a copy of thread 1's genome (due to the reduction),
                    // we'll replace it with the migrant. Note this is safe because the last
                    // iteration of the for loop did not involve thread 2, and we sync'd before that
                    (shared_memory + 1)->genome = *inbox(index);
                }
            }
            
            // Perform mutation
            operations.mutate(index, threadIdx.x, &thread_memory->genome, &thread_memory->rand_state);
            
            // Perform crosses
            if (threadIdx.x % 2 == 1) {
                shared_memory[threadIdx.x].genome = operations.cross(
                                                                     index,
                                                                     shared_memory[threadIdx.x - 1].genome,
                                                                     shared_memory[threadIdx.x].genome,
                                                                     &thread_memory->rand_state
                                                                     );
            }
        }
         */
    }
    
    template <typename Spawn, typename Evaluate, typename Mutate, typename Cross>
    void cudaCallSimulation(const size_t blocks, const size_t threadsPerBlock, genome_t * const inboxes, specification_t specification, size_t max_iterations, Spawn spawn, Evaluate evaluate, Mutate mutate, Cross cross) {
        unsigned bytesPerBlock = sizeof(thread_data) * threadsPerBlock;
        
        cudaSimulationKernel<<<blocks, threadsPerBlock, bytesPerBlock>>>(inboxes, specification, max_iterations, spawn, evaluate, mutate, cross);
    }
}


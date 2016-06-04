
#include "genetic_algorithm.hpp"
#include <stdlib.h>
#include <string.h>

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

namespace genetic_algorithm {
    // GPU TODO: For now, we'll just return a pointer to the global memory (w/ all of the best values)
    //           after it runs. We should figure out a better alternative to this on the GPU.
    float *simulation::run(specification_t specification, size_t max_iterations) {
        unsigned blocks = specification.kingdom_count * specification.island_count_per_kingdom;
        unsigned threadsPerBlock = specification.island_population;
        unsigned bytesPerBlock = sizeof(thread_data) * threadsPerBlock;
        
        // Emulating shared memory on the CPU so it can be programmed similiarly
        void *__all_block_shared_memory = malloc(bytesPerBlock * blocks);
        
        // Need enough global memory for an inbox on every island.
        void *global_memory = malloc(blocks * sizeof(genome_t));
        
        // Used to enumlate the CPU just waiting for all blocks on the GPU to complete with synchronize.
        bool *finished_blocks = (bool *)malloc(blocks * sizeof(bool));
        memset(finished_blocks, false, blocks * sizeof(bool));
        
        // Initialize each individual
        foreach_block({
            foreach_thread({
                thread_memory->genome = operations.spawn(index);
            });
        });
        
        // Helpers for accessing a specific inbox by index
#define mailboxes(cross_section) ((genome_t *)global_memory + (cross_section) * specification.kingdom_count)
#define inbox(index) (mailboxes((index).cross_section_index) + (index).kingdom_index)
        
        // This loop should be in the GPU code. The kernel will run independently
        // of the GPU until it converges on a solution. Further, this loop will not
        // require this sort of loop-over-data-check or malloc since each thread will
        // just check its own local variable. Once all threads finish, our kernel
        // finishes.
        
        // Note that the stopping condition will be check by every thread, but this
        // should be just as efficient as any alternative.
        for (size_t i = 0; i < max_iterations; i++) {
            
            // Evaluate each individual
            foreach_block({
                foreach_thread({
                    thread_memory->fitness = operations.evaluate(index, thread_memory->genome, mailboxes(index.cross_section_index));
                });
            });
            
            // Reduce to find the best individual, killing off subpar ones in the process
            foreach_block({
                // Perform reduction on genomes taking the one with the min/maximum fitness
                // Note that this almost follows a standard GPU reduction, just done on the CPU.
                // Unlike a standard reduction though, we reduce both *left* and *right* so that
                // subpar individuals are eliminated even if the fitnesses are alread arranged
                // in descending order. In the worst case, this will elimiate half of the
                // population replacing them with better individuals, more often the best ones.
                
                // Example reduction:
                // 1 2 3 4 5 4 3 2
                // 5 4 3 4|5 4 3 4 <- both sides will be the same at this point, max of each element
                // 5 4|5 4         <- now we leave the right side alone and do the same
                // 5|5             <- repeating again
                // 5 5 5 4 5 4 3 4 <- final result has largest value on left and elimiated the lowest
                
                for (size_t bound = threadsPerBlock >> 1; bound > 0; bound >>= 1) {
                    for (size_t threadIdx_x = 0; threadIdx_x < bound; threadIdx_x++) {
                        thread_data *thread_memory = &shared_memory[threadIdx_x];
                        thread_data *other_memory = thread_memory + bound;
                        
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
                }
            });
            
            // Perform migration of elite individual
            foreach_block({
                // Outward
                {
                    // Send updated elite genome to next island in kingdom
                    island_index next_index = ((island_index){
                        .kingdom_index = index.kingdom_index,
                        .cross_section_index = (index.cross_section_index + 1) % specification.island_count_per_kingdom
                    });
                    // Note that the leftmost genome in shared memory is the best since we reduced left
                    *inbox(next_index) = shared_memory->genome;
                }
                
                // Inward
                {
                    // Migrate elite individual from island into the population
                    // Since thread 2 has a copy of thread 1's genome (due to the reduction),
                    // we'll replace it with the migrant.
                    (shared_memory + 1)->genome = *inbox(index);
                }
            });
            
            // Print the elite fitness values.
            //                    foreach_block({
            //                        printf("k %zu:%zu\tg %f\tf %f\n",
            //                            index.kingdom_index, index.cross_section_index, shared_memory->genome, shared_memory->fitness);
            //                    });
            
            // Perform mutations
            foreach_block({
                foreach_thread({
                    operations.mutate(index, threadIdx_x, &thread_memory->genome);
                });
            });
            
            // Perform crosses
            foreach_block({
                for (size_t threadIdx_x = 2; threadIdx_x < threadsPerBlock; threadIdx_x += 2) {
                    shared_memory[threadIdx_x].genome = operations.cross(index,
                                                                         shared_memory[threadIdx_x].genome,
                                                                         shared_memory[threadIdx_x + 1].genome
                                                                         );
                }
            });
        }
        
        free(finished_blocks);
        free(__all_block_shared_memory);
        
        return mailboxes(0);
    }
};


#include "genetic_algorithm_cuda.cuh"

using namespace genetic_algorithm;

// Helpers for accessing a specific inbox by index
#define mailboxes(cross_section) (inboxes + (cross_section) * specification.kingdom_count)
#define inbox(index) (mailboxes((index).cross_section_index) + (index).kingdom_index)

__global__
void cudaSimulationKernel(genome_t *inboxes, specification_t specification, operations_t operations, size_t max_iterations) {
    extern __shared__ thread_data shared_memory[];
    thread_data *thread_memory = shared_memory + threadIdx.x;

    island_index index = (island_index){
        .kingdom_index = blockIdx.x / specification.island_count_per_kingdom,
        .cross_section_index = blockIdx.x % specification.island_count_per_kingdom
    };

    // Initialize individual
    thread_memory->genome = operations.spawn(index);
    
    
    for (size_t i = 0; i < max_iterations; i++) {
        // Evaluate individual
        thread_memory->fitness = operations.evaluate(index, thread_memory->genome, mailboxes(index.cross_section_index));

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
                island_index next_index = ((island_index){
                    .kingdom_index = index.kingdom_index,
                    .cross_section_index = (index.cross_section_index + 1) % specification.island_count_per_kingdom
                });
                // Note that this thread's genome in shared memory is the best since we reduced left
                *inbox(next_index) = thread_memory->genome;
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
        operations.mutate(index, threadIdx.x, &thread_memory->genome);

        // Perform crosses
        if (threadIdx.x % 2 == 1) {
            shared_memory[threadIdx.x].genome = operations.cross(
                index,
                shared_memory[threadIdx.x - 1].genome,
                shared_memory[threadIdx.x].genome
            );
        }
    }
}

void cudaCallSimulation(const size_t blocks, const size_t threadsPerBlock, genome_t *inboxes, specification_t specification, operations_t operations, size_t max_iterations) {
    unsigned bytesPerBlock = sizeof(thread_data) * threadsPerBlock;

    cudaSimulationKernel<<<blocks, threadsPerBlock, bytesPerBlock>>>(inboxes, specification, operations, max_iterations);
}

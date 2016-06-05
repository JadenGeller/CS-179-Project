#pragma once

#include <vector>
#include <stddef.h>
#include <cuda_runtime.h>
#include <nvfunctional>

// TODO: Potentially make generic using template parameters. This might hurt performance though
//       since we'd have to introduce another lambda.
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
    
    struct operations_t {
        // Return a random genome.
        // GPU TODO: How will we get entropy from the GPU? Will it have to be passed in,
        //           or can we generate it dynamically.
        nvstd::function<genome_t(island_index index)> spawn;
        
        // Evaluate the `test_genome` returning its fitness.
        // Note that `competitor_genomes` is an array of the best genomes in each kingdom,
        // ordered by kingdom, fomr `test_genomes` cross section of the world.
        nvstd::function<fitness_t(island_index index, genome_t test_genome, genome_t *competitor_genomes)> evaluate;
        
        // DECIDE: Should we allow the caller to determine whether or not its mutated like this?
        // Mutate a genome or leave it alone.
        nvstd::function<void(island_index index, size_t genome_index, genome_t *genome)> mutate;
        
        // DECIDE: Any reason to share the genome indices here? It seems inconsistent to not.
        // Cross two genomes.
        nvstd::function<genome_t(island_index index, genome_t genome_x, genome_t genome_y)> cross;
    };
    
    class simulation {
    private:
        operations_t operations;
        
    public:
        simulation(operations_t operations): operations(operations) { }
        void run(specification_t specification, size_t max_iterations, float *results);
    };
};

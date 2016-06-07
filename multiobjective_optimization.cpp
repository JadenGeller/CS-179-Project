#include "multiobjective_optimization.hpp"
#include <random>
#include <cstring>
#include "genetic_algorithm.hpp"

namespace multiobjective_optimization {
    using namespace genetic_algorithm;
    // ISSUE: CUDA FILES INCLUDED INTO MAIN
    void mathematical_optimization::compute(float *optimized_arguments, float *optimized_fitness) {

        /*
        // GPU TODO: We need to generate random numbers elsewhere.
        simulation simulation ({
            .spawn = [] __device__ (island_index index, curandState_t *rand_state) -> fitness_t {
                // TODO: Generalize for any problem?
                return 20 * curand_uniform(rand_state) - 10;
            },
            .evaluate = [this] __device__ (island_index index, genome_t test_genome, genome_t *competitor_genomes, curandState_t *rand_state) -> fitness_t {
                // GPU TODO: We need to be able to have an arbitrary number of arguments on the stack.
                //           We probably cannot dynamically allocate stack space in a CUDA lambda.
                //           Even if we could, it seems like a bad idea. We already have the array
                //           `competitor_genomes`. Too bad we can't tell it to swap a certain index
                //           out for a different one. Maybe if we made it an array of pointers???
                
                return 0;
//                float args[argument_count];
//                for (size_t i = 0; i < argument_count; i++) {
//                    if (i == index.kingdom_index) {
//                        args[i] = test_genome;
//                    } else {
//                        args[i] = competitor_genomes[i];
//                    }
//                }
//                
//                return functions[index.kingdom_index](args);
            },
            .mutate = [] __device__ (island_index index, size_t genome_index, genome_t *genome, curandState_t *rand_state) {
                if (genome_index == 0) return; // Leave the elite alone.
                if (curand_uniform(rand_state) > 0.5) {
                    // TODO: This is niave. Maybe use distance dependent mutation?
                    *genome *= -1.0 / (curand_uniform(rand_state) - 1.1);
                    *genome += -1.0 / (curand_uniform(rand_state) - 1.1);
                    *genome -= -1.0 / (curand_uniform(rand_state) - 1.1);
                }
            },
            .cross = [] __device__ (island_index index, genome_t genome_x, genome_t genome_y, curandState_t *rand_state) -> genome_t {
                // GPU TODO: Can we make this more parallel?
                float scale = curand_uniform(rand_state);
                // TODO: This is niave. Does it really represent a good cross of genes?
                return scale * genome_x + (1 - scale) * genome_y;
            }
        });
         */
        
        
        simulate(optimized_arguments, {
            .max_iterations = 1000,
            .kingdom_count = functions.size(),
            .island_count_per_kingdom = 10,
            .island_population = 50
        },
                 [] __device__ (island_index index, curandState_t *rand_state) -> fitness_t {
                     // TODO: Generalize for any problem?
                     return 20 * curand_uniform(rand_state) - 10;
                 },
                 [this] __device__ (island_index index, genome_t test_genome, genome_t *competitor_genomes, curandState_t *rand_state) -> fitness_t {
                     // GPU TODO: We need to be able to have an arbitrary number of arguments on the stack.
                     //           We probably cannot dynamically allocate stack space in a CUDA lambda.
                     //           Even if we could, it seems like a bad idea. We already have the array
                     //           `competitor_genomes`. Too bad we can't tell it to swap a certain index
                     //           out for a different one. Maybe if we made it an array of pointers???
                     
                     return 0;
                     //                float args[argument_count];
                     //                for (size_t i = 0; i < argument_count; i++) {
                     //                    if (i == index.kingdom_index) {
                     //                        args[i] = test_genome;
                     //                    } else {
                     //                        args[i] = competitor_genomes[i];
                     //                    }
                     //                }
                     //                
                     //                return functions[index.kingdom_index](args);
                 },
                 [] __device__ (island_index index, size_t genome_index, genome_t *genome, curandState_t *rand_state) {
                     if (genome_index == 0) return; // Leave the elite alone.
                     if (curand_uniform(rand_state) > 0.5) {
                         // TODO: This is niave. Maybe use distance dependent mutation?
                         *genome *= -1.0 / (curand_uniform(rand_state) - 1.1);
                         *genome += -1.0 / (curand_uniform(rand_state) - 1.1);
                         *genome -= -1.0 / (curand_uniform(rand_state) - 1.1);
                     }
                 },
                 [] __device__ (island_index index, genome_t genome_x, genome_t genome_y, curandState_t *rand_state) -> genome_t {
                     // GPU TODO: Can we make this more parallel?
                     float scale = curand_uniform(rand_state);
                     // TODO: This is niave. Does it really represent a good cross of genes?
                     return scale * genome_x + (1 - scale) * genome_y;
                 }
        );
        
        if (optimized_fitness != NULL) {
            for (size_t i = 0; i < functions.size(); i++) {
                optimized_fitness[i] = functions[i](optimized_arguments);
            }
        }
    }
};

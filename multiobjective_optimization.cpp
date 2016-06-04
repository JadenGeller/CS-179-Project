#include "multiobjective_optimization.hpp"
#include <random>
#include <cstring>

namespace multiobjective_optimization {
    using namespace genetic_algorithm;
    
    void mathematical_optimization::compute(float *optimized_arguments, float *optimized_fitness) {
        
        // GPU TODO: We need to generate random numbers elsewhere.
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0, 1);
        
        simulation simulation ({
            .spawn = [&dist, &gen](island_index index) -> fitness_t {
                // TODO: Generalize for any problem?
                return 20 * dist(gen) - 10;
            },
            .evaluate = [this](island_index index, genome_t test_genome, genome_t *competitor_genomes) -> fitness_t {
                // GPU TODO: We need to be able to have an arbitrary number of arguments on the stack.
                //           We probably cannot dynamically allocate stack space in a CUDA lambda.
                //           Even if we could, it seems like a bad idea. We already have the array
                //           `competitor_genomes`. Too bad we can't tell it to swap a certain index
                //           out for a different one. Maybe if we made it an array of pointers???
                
                float args[argument_count];
                for (size_t i = 0; i < argument_count; i++) {
                    if (i == index.kingdom_index) {
                        args[i] = test_genome;
                    } else {
                        args[i] = competitor_genomes[i];
                    }
                }
                
                return functions[index.kingdom_index](args);
            },
            .mutate = [&dist, &gen](island_index index, size_t genome_index, genome_t *genome) {
                if (genome_index == 0) return; // Leave the elite alone.
                if (dist(gen) > 0.5) {
                    // TODO: This is niave. Maybe use distance dependent mutation?
                    *genome *= -1.0 / (dist(gen) - 1.1);
                    *genome += -1.0 / (dist(gen) - 1.1);
                    *genome -= -1.0 / (dist(gen) - 1.1);
                }
            },
            .cross = [&dist, &gen](island_index index, genome_t genome_x, genome_t genome_y) -> genome_t {
                // GPU TODO: Can we make this more parallel?
                float scale = dist(gen);
                // TODO: This is niave. Does it really represent a good cross of genes?
                return scale * genome_x + (1 - scale) * genome_y;
            }
        });
        
        float *results = simulation.run({
            .kingdom_count = functions.size(),
            .island_count_per_kingdom = 10,
            .island_population = 50,
        }, 1000);
        
        if (results != NULL) memcpy(optimized_arguments, results, argument_count * sizeof(float));
        
        if (optimized_fitness != NULL) {
            for (size_t i = 0; i < functions.size(); i++) {
                optimized_fitness[i] = functions[i](results);
            }
        }
        free(results);
        
        return ;
    }
};

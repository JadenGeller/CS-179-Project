#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>
#include <random>

/*
Below is an example of a 3 kindom setup where each kindom contains 4 islands.
Each kingdom works to optimize a certain objective by controlling a specific
variable while the other variables are fixed by the other islands (who are
themselves trying to optimize some other objective).

+-------------+  +-------------+  +-------------+
| +---+ +---+ |  | +---+ +---+ |  | +---+ +---+ |
| |1:1|>|1:2| |  | |2:1|>|2:2| |  | |3:1|>|3:2| |
| +---+ +---+ |  | +---+ +---+ |  | +---+ +---+ |
|   ^     v   |  |   ^     v   |  |   ^     v   |
| +---+ +---+ |  | +---+ +---+ |  | +---+ +---+ |
| |1:4|<|1:3| |  | |2:4|<|2:3| |  | |3:4|<|3:3| |
| +---+ +---+ |  | +---+ +---+ |  | +---+ +---+ |
+-------------+  +-------------+  +-------------+

Note the arrows indiate migration patterns within the kindgom. The best
individual on the nth island is copied into the migratory outbox this island.
Then, once the ((n + 1) % island_count_per_kindgom island)th island in the same
kingdom is ready to accept an immigrant, it copies the individual into its
population. This is important as it allows communication of best strategies
between islands within a kingdom.

Islands of other kingdoms are able to spy on the migratory outbox of islands
with their same within-kingdom index. For example, island 2:1 can observe
the migration our of 1:1 and 3:1 and thus learn about their best individuals.
This is important since each island must know the current strategy of the other
kingdoms in order to know how to optimize their objective.
*/

/* Shared Memory
 * -------------
 * + For each thread...
 *   + Genomes
 *   + Fitness scores
 * + It's not clear how we should optimally organize given the access pattern.
 *   + We'll just use a struct for now, and consider optimizing later.
 *
 * Global Memory
 * -------------
 * + Inboxes 
 *   + Organized by island identifier, then kingdom
 *     + Optimizes lookup speed of common case
 *     + Ex. { 1:1, 2:1, 3:1, 1:2, 2:2, 3:2, 1:3, 2:3, 3:3, 1:4, 2:4, 3:4;, 4:4 }
 */

// Runs code on each block sequentially, declaring a few local variables in the process.
// Requires that `blocks` and `threadsPerBlock` be defined in the local scope. 
#define foreach_block(braced_code) ({\
    unsigned blockDim_x = threadsPerBlock;\
    for (unsigned blockIdx_x = 0; blockIdx_x < blocks; blockIdx_x++) {\
        /* For CPU version, only run if we shouldn't have exited the loop */\
        if (finished_blocks[blockIdx_x]) continue;\
        /* Compute the location of *our* shared memory */\
        thread_data *shared_memory = (thread_data *)((char *)__all_block_shared_memory + blockIdx_x * bytesPerBlock);\
        island_index index = (island_index){\
            .kingdom_index = blockIdx_x / specification.island_count_per_kingdom,\
            .cross_section_index = blockIdx_x % specification.island_count_per_kingdom\
        };\
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
    
    class simulation {
        public:
            struct specification_t {
                size_t kingdom_count;
                size_t island_count_per_kingdom;
                size_t island_population;
            };
            
            struct operations_t {
                // Return a random genome.
                // GPU TODO: How will we get entropy from the GPU? Will it have to be passed in,
                //           or can we generate it dynamically.
                std::function<genome_t(island_index index)> spawn;
                
                // Evaluate the `test_genome` returning its fitness.
                // Note that `competitor_genomes` is an array of the best genomes in each kingdom,
                // ordered by kingdom, fomr `test_genomes` cross section of the world.
                std::function<fitness_t(island_index index, genome_t test_genome, genome_t *competitor_genomes)> evaluate;
                
                // DECIDE: Should we allow the caller to determine whether or not its mutated like this?
                // Mutate a genome or leave it alone.
                std::function<void(island_index index, size_t genome_index, genome_t *genome)> mutate;
                
                // DECIDE: Any reason to share the genome indices here? It seems inconsistent to not.
                // Cross two genomes.
                std::function<genome_t(island_index index, genome_t genome_x, genome_t genome_y)> cross;
        };
            
        private:
            operations_t operations;
            
        public:
            simulation(operations_t operations): operations(operations) { }
            
            // GPU TODO: For now, we'll just return a pointer to the global memory (w/ all of the best values)
            //           after it runs. We should figure out a better alternative to this on the GPU.
            float *run(specification_t specification, size_t max_iterations) {
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
                for (int i = 0; i < max_iterations; i++) {
                    
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
};

// inspired by:
// http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.36.5013&rep=rep1&type=pdf
namespace multiobjective_optimization {
   
    // TODO: Also support maximization
//    enum optimization_type {
//        minimize,
//        maximize
//    };

    // TODO: Maybe make a general optimzation type
//    template <typename X, typename F>
//    class optimzation {
//        private:
//            virtual bool is_increasing(X from_value, X to_value) {
//                printf("Abstract class!\n");
//                exit(1);
//            }
//        public:
//            // TODO: Potentially support constraints
//        
//            F apply(X argument) {
//                
//            }    
//    }
    
    class mathematical_optimization/*: public optimization*/ {
        public:
            size_t argument_count; // TODO: Make read only?
            std::vector<std::function<float(float *)>> functions;        
            
            void compute(float *optimized_arguments, float *optimized_fitness) {
                using namespace genetic_algorithm;

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
                        for (int i = 0; i < argument_count; i++) {
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
                            genome_t before = *genome;
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
};

int main(int argc, char *argv[]) {
    using namespace multiobjective_optimization;
    
    // TODO: Experiment with other optimization problems.
    mathematical_optimization problem = {
        .argument_count = 2,
        .functions = {
            [](float *args) -> float { return pow(args[0] - 1, 2) + pow(args[0] - args[1], 2); },
            [](float *args) -> float { return pow(args[0] - 3, 2) + pow(args[0] - args[1], 2); }	
        }
    };
   
    float optimized_arguments[2];
    float optimized_fitness[2];
    problem.compute(optimized_arguments, optimized_fitness);
    printf("optimal: (");
    for (float *x = optimized_arguments; x < optimized_arguments + problem.argument_count; x++) {
        if (x != optimized_arguments) printf(", ");
        printf("%f", *x);
    }
    printf(") => {");
    for (float *y = optimized_fitness; y < optimized_fitness + problem.functions.size(); y++) {
        if (y != optimized_fitness) printf(", ");
        printf("%f", *y);
    }
    printf("}\n");
    
    return 0;
}

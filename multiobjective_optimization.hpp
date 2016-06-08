#pragma once

#include <vector>
#include <cstring>
#include "genetic_algorithm.hpp"

// inspired by:
// http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.36.5013&rep=rep1&type=pdf

// Evaluates to the count of a statically allocated C array
#define countof(arr) (sizeof(arr)/sizeof(arr[0]))

// Used to specify a function to optimize. The first parameter is the index of the
// function and the second is the result. There's an implict `args` array that
// contains the arguments of the function, and it can be used within the computation
// of the value.
#define func(function_index, value) case function_index: result = value; break;

#define optimize(iterations, arguments, fitness, functions) ({\
    using namespace genetic_algorithm;\
    const size_t argument_count = countof(arguments);\
    const size_t function_count = countof(fitness);\
    simulate(\
        /* results: */ arguments,\
        /* specifications: */ {\
            .max_iterations = iterations,\
            .kingdom_count = function_count,\
            .island_count_per_kingdom = 10,\
            .island_population = 50\
        },\
        /* spawn: */ [] __device__ (island_index index, curandState_t *rand_state) -> fitness_t {\
            return 20 * curand_uniform(rand_state) - 10;\
        },\
        /* evaluate: */ [argument_count] __device__ (island_index index, genome_t test_genome, genome_t *competitor_genomes,curandState_t *rand_state) -> fitness_t {\
            float args[argument_count];\
            for (size_t i = 0; i < argument_count; i++) {\
                if (i == index.kingdom_index) {\
                    args[i] = test_genome;\
                } else {\
                    args[i] = competitor_genomes[i];\
                }\
            }\
            float result = -1;\
            switch(index.kingdom_index) functions\
            return result;\
        },\
        /* mutate: */ [] __device__ (island_index index, size_t genome_index, genome_t *genome, curandState_t *rand_state) {\
            if (genome_index == 0) return; /* Leave the elite alone. */\
            if (curand_uniform(rand_state) > 0.5) {\
                *genome *= -1.0 / (curand_uniform(rand_state) - 1.1);\
                *genome += -1.0 / (curand_uniform(rand_state) - 1.1);\
                *genome -= -1.0 / (curand_uniform(rand_state) - 1.1);\
            }\
        },\
        /* cross: */ [] __device__ (island_index index, genome_t genome_x, genome_t genome_y, curandState_t *rand_state) -> genome_t {\
            float scale = curand_uniform(rand_state);\
            return scale * genome_x + (1 - scale) * genome_y;\
        }\
    );\
    for (size_t f = 0; f < function_count; f++) {\
        float result = -1;\
        float *args = arguments;\
        switch(f) functions\
        fitness[f] = result;\
    }\
    (void)0;\
})


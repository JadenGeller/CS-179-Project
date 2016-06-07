#pragma once

#include <vector>
#include <nvfunctional>
#include <cstring>
#include "genetic_algorithm.hpp"

// inspired by:
// http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.36.5013&rep=rep1&type=pdf

#define func(function_index, value) case function_index: result = value; break;

#define ArgC(arr) (sizeof(arr)/sizeof(arr[0]))

#define optimize(arguments, FunC, functions) ({\
    using namespace genetic_algorithm;\
    const size_t argument_count = ArgC(arguments);\
    simulate(\
        /* results: */ arguments,\
        /* specifications: */ {\
            .max_iterations = 1000,\
            .kingdom_count = FunC,\
            .island_count_per_kingdom = 10,\
            .island_population = 50\
        },\
        /* spawn: */ [] __device__ (island_index index, curandState_t *rand_state) -> fitness_t {\
            /* TODO: Generalize for any problem? */\
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
                /* TODO: This is niave. Maybe use distance dependent mutation? */\
                *genome *= -1.0 / (curand_uniform(rand_state) - 1.1);\
                *genome += -1.0 / (curand_uniform(rand_state) - 1.1);\
                *genome -= -1.0 / (curand_uniform(rand_state) - 1.1);\
            }\
        },\
        /* cross: */ [] __device__ (island_index index, genome_t genome_x, genome_t genome_y, curandState_t *rand_state) -> genome_t {\
            /* GPU TODO: Can we make this more parallel? */\
            float scale = curand_uniform(rand_state);\
            /* TODO: This is niave. Does it really represent a good cross of genes? */\
            return scale * genome_x + (1 - scale) * genome_y;\
        }\
    );\
    (void)0;\
})


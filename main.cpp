
#include "multiobjective_optimization.hpp"
#include <cmath>

int main(int argc, char *argv[]) {

    float optimized_arguments[2];
    float optimized_fitness[2];
    optimize(1000000, optimized_arguments, optimized_fitness, {
        func(0, args[0] * (args[0] - args[1] - 2));
        func(1, args[1] * (args[1] - args[0] - 8));
    });
    
    printf("optimal: (");
    for (float *x = optimized_arguments; x < optimized_arguments + countof(optimized_arguments); x++) {
        if (x != optimized_arguments) printf(", ");
        printf("%f", *x);
    }
    printf(") => {");
    for (float *y = optimized_fitness; y < optimized_fitness + countof(optimized_fitness); y++) {
        if (y != optimized_fitness) printf(", ");
        printf("%f", *y);
    }
    printf("}\n");
    
    return 0;
}

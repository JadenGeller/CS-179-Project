
#include "multiobjective_optimization.hpp"
#include <cmath>

int main(int argc, char *argv[]) {
    
    // TODO: Experiment with other optimization problems.
//    optimize(2, 2, {
//        case 0: result = pow(args[0] - 1, 2) + pow(args[0] - args[1], 2);
//        case 1: result = pow(args[0] - 3, 2) + pow(args[0] - args[1], 2);
//    });
    

    float optimized_arguments[2];
    float optimized_fitness[2];
    optimize(optimized_arguments, optimized_fitness, {
        func(0, pow(args[0] - 1, 2) + pow(args[0] - args[1], 2));
        func(1, pow(args[0] - 3, 2) + pow(args[0] - args[1], 2));
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

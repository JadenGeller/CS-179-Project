#pragma once

#include <vector>
#include <functional>

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
        
        void compute(float *optimized_arguments, float *optimized_fitness);
    };
};


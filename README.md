# Parallel Multi-Objective Optimization via Genetic Algorithm

## Abstract

I implemented a massively parallel genetic algorithm framework that resembles a modified version of the island model. The framework is built atop CUDA 7.5 and utilizes the experimental device lambda feature. In implementing this, I discovered undocumented current limitations of device lambdas and found novel workarounds to express the API. I used this framework to compute Nash equilibrium by coevolving game-playing strategies between interacting agents.

## Background and Design

### Game Theroy

Game theory studies the interactions of rational decision-makers. We can simulate various games using evolutionary strategies to determine the equilibrium behavior, or Nash equilibrium, of the players.

Consider the game following game in which two countries choose a tarrif rate to impose on imports. Each function `u1` and `u2` represents the payoff of a given country as a function of the rates `t1` and `t2` imposed.

```
u1(t1, t2) = -t1(t1 - t2 - 2)
u2(t1, t2) = -t2(t2 - t1 - 8)
```

The Nash equilbiria of this strategic game is the equilibria where neither country would increase their payoff by increasing their tariff. In this particular game, analytical computations calculate that the unique Nash equilbrium is `(t1*, t2*) = (4, 6)`. We will determine whether similiar results can be achieved using our genetic algorithm framework.

### Genetic Algorithms

Genetic algorithms are used to solve optimization problems using a process similar to natural selection. This class of algorithms generates and measures the fitness of massive populations across many iterations, so it’s a promising candidate for GPU acceleration since each gene's fitness can be computed on an individual thread.

We will use the methods described in "Nash Genetic Algorithms" [[1]](#references) to compute Nash equilibrium via genetic algorithm, though we will modify it for better parallelization. Note that Nash equilibrium is computed by having each player optimizing their fitness function while the other parameters are fixed by the competing players. 

In this project, the genetic representation of a given agent corresponds with a particular strategy. Our implementation will isolate subpopulations within each multiprocessor “island” since this eliminates the slow process of inter-multiprocessor communication. Further, this allows us to minimize copy operations into slower memory. These “islands” will compete and evolve independently except for the occasional asynchronous “migration” of agent between island.

The migrations will transfer the most fit individuals between islands in order to provide better competitors to compete against, therefore getting closer to the Nash equilbrium. Random mutations and crossing of genes will make sure we reach the global optimum rather than simply a local optimum.

## Previous Work

### GPU–Accelerated Genetic Algorithms

Researchers have shown that CUDA implementations of genetic algorithms can introduce significant speedups over their CPU counterparts. Specifically, researchers have proposed an “island”-based model for fast massively-parallel computation [[3]](#references). Competing models involve generating the populations on the CPU and running the fitness tests of the GPU, but this doesn’t have quite as large a speedup due to expensive copying operations. It has been shown that the parallel island model produces solutions with similar quality to the CPU based algorithm despite the modifications.

### Co-Evolvability of Games

Researchers have studied the ability of coevolutionary genetic algorithms to evolve game-playing strategies [[4]](#references). Some studies have found that Nash equilibrium can be found by genetic algorithms. If this is feasible for all games, accelerated evolutionary algorithms might provide a fast way to estimate mixed Nash equilibrium, which is sometimes too computationally complex to feasibly compute analytically. It is not clear that this sort of computation is always possible though. Some studies note that collusion develops among agents that coevolve, and thus the Nash equilibrium is not always found. It seems that the feasibility of its computation might depend on the game type, the fitness function, and the evolutionary technique. 

## Implementation

### Genetic Algorithm

The genetic algorithm framework is implemented using CUDA device lambdas so any arbitrary genetic algorithm can be encoded in the framework without having to write a separate kernel. The framework supports specification of `spawn`, `evaluate`, `mutate`, and `cross` functions to specify the algorithm. Each function takes in a `curandState_t` that has been seeded specifically for that thread so that randomized behavior is easily implemented. Note that we seed each thread with its thread id. It is sufficient to do this since we only care that each thread has independent random numbers, but we do not care if the program acts randomly between invocations.

The `simulate` function also takes in a specification parameter that indicates how the algorithm should run, including the number of iterations. Further, the developer can specify the `kingdom_count`, the `island_count_per_kingdom`, and the `island_population`. These parameters require some explaning. The island model typically involves using a separate block for independent evolution of each population. We similarly regard each block as an island, but extend to model to include the concept of a "kingdom". A kingdom is an alliance of islands that is acting to achieve the same goal. In our implementation, each kingdom works together to optimize a separate criteria of the nash equilbrium. Each subpopulation within the kingdom simplify ensures genetic diversity is maintained. Migration occurs interkingdom in a ring between islands in that kingdom and intrakingdom in a ring between islands of the same cross-section of different kingdoms. Note that since our genome is represent as a floating point number, we can atomically mutate it in global memory and thus can asynchronously migrate genomes between islands without corruption.

Our genetic algorithm framework does not include a `selection` operator as is typical. This is because we take advantage of the parrallel nature to implement a binary reduction that computes the individual with the greatest fitness. During the reduction, we compare two genes and elimite the less fit one and replace it with a copy of the more fit one. In the process, it wipes out individuals with the worst fitness---approximately 50% of the population depending on the distribution of individuals within the islands. Though it seems odd to elimiate individuals of the population based partially on placement, it allows us to optimize the reduction to perform very quickly.

After evaluation, selection, and migration, we cross and mutate the individuals before repeating the process. This evolution loop will continue until we reach the desired number of iterations and then will stop. The kernel will finish once all populations have been evolved to the desired number of iterations. Note that there is no syncronization between each island by design as to ensure speedy performance.

### Multi-objective Optimization

The multi-objective optimization layer is written on top of the genetic algorithm framework. For reasons we will discuss later, it's API is exposed as a macro with a special syntax for specifying the optimization problem. Specifically, it is implemented to compute the Nash equilbirum of a set of `N` equations each with `N` arguments. Each player's controls the `i`th argument and uses the `i`th equation as its fitness function. Here is the encoding of the tarrif problem described above:

```c
    float optimized_arguments[2];
    float optimized_fitness[2];
    optimize(10000000, optimized_arguments, optimized_fitness, {
        func(0, args[0] * (args[0] - args[1] - 2));
        func(1, args[1] * (args[1] - args[0] - 8));
    });
```

The optimize macro takes a few parameters. First, the number of generations to evolve the population before returning the result. Next, statically allocated arrays that will, after running, contain the optimized arguments and the resulting optimized fitnesses. Note that it determines the number of parameters from the length of these arrays, and it is undefined behavior to have a different number of functions than should be expected by the length of the arrays. Finally, the last argument is a domain-specific language that describes each function. The syntax must use the `func` macro, which takes the function index as the first argument and its defintion as the second argument. Note that the second argument may use the implicitly defined `args` array to obtain the arguments of the function. We will discuss why this particular syntax was used later in the report.

Once run, this will call the kernel and wait for the specified number of generations to complete before copying the result into the statically allocated arrays.

## Results

The `main.c` file encodes the problem to be optimized. The program can be compiled with the `make` command and run with `./main`. If it is desired that the program is timed, simply run `time ./main`.

As previously discussed, the analytical solution to this problem is `(4, 6)`. Let's compare that to the CPU and GPU computed result. We will run both the CPU and GPU code with `island_count_per_kingdom = 10`, `island_population = 500`, and *one thousand iterations*.

| Hardware | Optimized Arguments  | Optimized Fitness        | Runtime  |
|----------|----------------------|--------------------------|----------|
| CPU      | (3.998751, 6.000210) | {-16.000841, -35.992508} | 2,540 ms |
| GPU      | (5.818570, 8.769202) | {-28.805599, -44.278927} |    83 ms |

Clearly the GPU run much more quickly but the produced less accurate results. We will increase the number of iterations to *one-hundred thousand iterations* and compare again.

| Hardware | Optimized Arguments  | Optimized Fitness        | Runtime    |
|----------|----------------------|--------------------------|------------|
| CPU      | (3.999207, 5.998609) | {-15.994436, -35.995243} | 249,000 ms |
| GPU      | (5.045824, 6.620116) | {-18.035248, -42.538933} |     737 ms |

With an increased number of iterations, we see that the GPU result becomes much closer to the analytical solution and still runs very, very quickly---much faster than the CPU runtime with 100x less iterations. Also, it is clear that the CPU algorithm becomes infeasible to run as the number of generations increase, a compelling argument for GPU parallelization of this sort of problem. Since genetic algorithms require simulating many individuals at once, all doing the same thing, we see immense speedups between the CPU and GPU version.

### Accuracy

It's important to recognize the much lower accuracy of the GPU output compared to the CPU output. This is something that is not fully understood, and would benefit from greater research. One potential contributing factor is quality of random number generation. The CUDA documentation warns that generating random numbers with separate `curandState_t`s on each thread could potentially result in correlated random numbers. It is unclear if that might be the cause, but that is definitely something that ought to be looked into. If this were to be solved, the GPU impelementation would be clearly preferable for any sort of parallelizable genetic algorithm.

## Challenges

### Experimental CUDA Device Lambda

CUDA device lambdas were introduced in CUDA 7.5 as an experimental feature. As such, they are very poorly documented and have significant limitations that are often only discovered after implementation. Working through these issues was a very time-consuming part of the project.

CUDA defines a `nvstd::function` wrapper for resprenting the types of lambdas. Previously, the codebase followed a much more object-oriented model. A genetical algorithm simulation object was initialized using a struct containing the operations used in the algorithm:

```c
struct operations_t {
    // Return a random genome.
    nvstd::function<genome_t(island_index index, curandState_t *rand_state)> spawn;
    
    // Evaluate the `test_genome` returning its fitness.
    nvstd::function<fitness_t(island_index index, genome_t test_genome, genome_t *competitor_genomes, curandState_t *rand_state)> evaluate;
    
    // Mutate a genome or leave it alone.
    nvstd::function<void(island_index index, size_t genome_index, genome_t *genome, curandState_t *rand_state)> mutate;
    
    // Cross two genomes.
    nvstd::function<genome_t(island_index index, genome_t genome_x, genome_t genome_y, curandState_t *rand_state)> cross;
};
```

The `nvstd::function` type conveniently provided documentation and type-saftey regarding the types of the lambdas. After debugging memory issues for many hours, it was discovered that `nvstd::function` does NOT support device lambdas, only host lambdas, and this is not documented anywhere. Further, you may NOT store a device lambda in a struct as pass that as a function argument; you must DIRECTLY pass the lambda as a function argument or CUDA will not copy the lambda onto the device memory. These issues took ages to debug and significantly decreased the friendliness of the API. Now, the API is represented as a single top-level function that takes in all operations as an arguments. Further, the type of the arguments can NOT be represented, and template parameters must be used. This means that type errors made at the call-site propogate into the implementation in a way that is very difficult to debug. Unfortunately, this is the way it must be implemented though with the way CUDA device lambdas currently work. Now, the function signature is as follows:

```c
template <typename Spawn, typename Evaluate, typename Mutate, typename Cross>
static void simulate(float *results, specification_t specification, Spawn spawn, Evaluate evaluate, Mutate mutate, Cross cross) { ... }
```

### Template Metaprogramming

As mentioned, templates had to be used to represent the types of the lambda. If you are not familiar with templates in C++, they allow for the parameterization of a class or function. Unfortunately, this requires that *implementation* of the function be declared in the *header* file so that the code can be specialized. As such, much of the codebase is written in header files rather than a more clean header/implementation split. Further, this means that any codebase that uses the API much be compiled with NVVC; uses of the API don't simply need to link with the API, but actually must include an compile the API.

In faced quite a few *compiler* **crashes** while building the API. Since the lambda feature is experimental, there are currently bugs that cause compiler assertion failures in undocumented situations. The compiler does not report what went wrong, so it was quite a challenge to debug what caused the issue. This involved reducing the issue to the smallest base case where I observed this behavior and trying to find a workaround. I probably had to work through over a dozen major compiler crashes to get my code to compile over the course of the project, and that is in addition to normal type-error debugging.

### Macros

One limitation I discovered is that CUDA device lambdas cannot capture other device lambdas. This means that though multi-objective optimization can be done with the genetic algorithm framework, a multi-objective optimization framework that uses lambdas to represent the functions to be optimized cannot be built. I spent many hours trying to find a work around, even attempting to define the functions to be optimized in global functions and pass them as function pointers, but each approach I tried failed, usually due to undocumented limitations of the feature or crashes that exist in the current experimental implementation of device lambdas, expecially when used in conjunction with templates.

I finally worked around this issue by using macros. Since the multi-objective optimization project can be specified manually with the genetic algorithm framework, a macro could transform a problem into the required representation. As such, the `optimize` function was represented as a macro. Behind the scence, it takes the function representation that the developer provides and pastes it into a switch statement that switches on the index of the function to be executed. The `func` macro simply encodes a given case of the switch statement, setting a result local variable to the value and breaking. While this is not an ideal API, it does provide an easy to use, abstracted way to represent Nash equilibrium problems.

## Future Directions

TODO: Talk about seeding
TODO: Talk about selection
TODO: Talk about generalizing more
TODO: Talk about stopping conditions and making sure an island doesn't finish way before the others
TODO: Talk about focused on CUDA lambdas and parallelization
TODO: Talk about mutation, cross, etc.
TODO: Talk about constraints, maximization, etc.

## Installation and Usage

## Performance Analysis

### Nash Equilibrium Computation

TODO (results too)

## Conclusion

TODO

## References

1. [Nash Genetic Algorithms : examples and applications](http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=870339)
2. [Examples and exercises on finding Nash equilibria of two-player games using best response functions](https://www.economics.utoronto.ca/osborne/2x3/tutorial/NEIEX.HTM)
3. [GPU-based Acceleration of the Genetic Algorithm](http://www.gpgpgpu.com/gecco2009/7.pdf)
4. [Co-Evolvability of Games in Coevolutionary Genetic Algorithms](http://delivery.acm.org/10.1145/1580000/1570208/p1869-lin.pdf?ip=131.215.158.223&id=1570208&acc=ACTIVE%20SERVICE&key=1FCBABC0A756505B%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&CFID=626681181&CFTOKEN=86827508&__acm__=1465302757_bd6a01c0e44c7266dd7ac4a5f67008b7)

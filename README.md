# Parallel Multi-Objective Optimization via Genetic Algorithm

## Abstract

I implemented a massively parallel genetic algorithm framework that resembles a modified version of the island model. The framework is built atop CUDA 7.5 and utilizes the experimental device lambda feature. I used this framework to compute Nash equilibrium by coevolving game-playing strategies between interacting agents.

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

We will use the methods described in "Nash Genetic Algorithms" [[1]](#references) to compute Nash equilibrium via genetic algorithm, though we will modify it for better parallelization. Note that Nash equilibrium is computed by having each player optimizing their fitness function while the other parameters are fixed by the other players. 

In this project, the genetic representation of a given agent corresponds with a particular strategy.

Our implementation will isolate subpopulations within each multiprocessor “island” since this eliminates the slow process of inter-multiprocessor. Further, this allows us to minimize copy operations into slower memory. These “islands” will compete and evolve independently except for the occasional asynchronous “migration” of agent between island.

## Previous Work

### GPU–Accelerated Genetic Algorithms

Researchers have shown that CUDA implementations of genetic algorithms can introduce significant speedups over their CPU counterparts. Specifically, researchers have proposed an “island”-based model for fast massively-parallel computation [[3]](#references). Competing models involve generating the populations on the CPU and running the fitness tests of the GPU, but this doesn’t have quite as large a speedup due to expensive copying operations. It has been shown that the parallel island model produces solutions with similar quality to the CPU based algorithm despite the modifications.

### Co-Evolvability of Games

Researchers have studied the ability of coevolutionary genetic algorithms to evolve game-playing strategies [[4]](#references). Some studies have found that Nash equilibrium can be found by genetic algorithms. If this is feasible for all games, accelerated evolutionary algorithms might provide a fast way to estimate mixed Nash equilibrium, which is sometimes too computationally complex to feasibly compute analytically. It is not clear that this sort of computation is always possible though. Some studies note that collusion develops among agents that coevolve, and thus the Nash equilibrium is not always found. It seems that the feasibility of its computation might depend on the game type, the fitness function, and the evolutionary technique. 

## Implementation

### Genetic Algorithm

TODO

### Multi-objective Optimization

TODO

## Challenges

### Algorithmic Design

TODO

### Experimental CUDA Device Lambda

TODO

Blah [[1]](#references)

### Template Metaprogramming

TODO

## Future Directions

TODO: Talk about focused on CUDA lambdas and parallelization
TODO: Talk about mutation, cross, etc.
TODO: Talk about constraints, maximization, etc.

## Installation and Usage

## Results

### Differences Between CPU and GPU Output

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

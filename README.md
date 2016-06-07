# Parallel Multi-Objective Optimization via Genetic Algorithm

## Abstract

I implemented a massively parallel genetic algorithm framework that resembles a modified version of the island model. The framework is built atop CUDA 7.5 and utilizes the experimental device lambda feature. I used this framework to compute Nash equilibrium by coevolving game-playing strategies between interacting agents.

## Background

### Game Theroy

Game theory studies the interactions of rational decision-makers. We can simulate various games using evolutionary strategies to determine the equilibrium behavior, or Nash equilibrium, of the players.

Consider the game following game in which two countries choose a tarrif rate to impose on imports. Each function `u1` and `u2` represents the payoff of a given country as a function of the rates `t1` and `t2` imposed.

```
u1(t1, t2) = -t1(t1 - t2 - 2)
u2(t1, t2) = -t2(t2 - t1 - 8)
```

The Nash equilbiria of this strategic game is the equilibria where neither country would increase their payoff by increasing their tariff. In this particular game, analytical computations calculate that the unique Nash equilbrium is `(t1*, t2*) = (4, 6)`. We will determine whether similiar results can be achieved using our genetic algorithm framework.

## Installation and Usage

## Design

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

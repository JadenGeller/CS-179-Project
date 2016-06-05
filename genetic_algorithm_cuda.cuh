#pragma once

#include "genetic_algorithm.hpp"

void cudaCallSimulation(const size_t blocks, const size_t threadsPerBlock, genetic_algorithm::genome_t *inboxes, genetic_algorithm::specification_t specification, genetic_algorithm::operations_t operations, size_t max_iterations);

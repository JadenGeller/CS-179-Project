//--------------------------------------------------------------------------
// TA_Utilities.cpp
// Allow a shared computer to run smoothly when it is being used
// by students in a CUDA GPU programming course.
//
// TA_Utilities.cpp/hpp provide functions that programatically limit
// the execution time of the function and select the GPU with the 
// lowest temperature to use for kernel calls.
//
// Author: Jordan Bonilla - 4/6/16
//--------------------------------------------------------------------------

#include "ta_utilities.hpp"

#include <unistd.h> // sleep, fork, getpid
#include <signal.h> // kill
#include <cstdio> // printf
#include <stdlib.h> // popen, pclose, atof, fread
#include <cuda_runtime.h> // cudaGetDeviceCount, cudaSetDevice
#define GB_PER_B 1.0e-9 // GigaBytes per Byte
#define DEBUGG 0 // Set to 1 to enable extra print statements

/*
Modified from:
http://stackoverflow.com/questions/14038589/
what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
*/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

namespace TA_Utilities
{
  /* Select the least utilized GPU on this system. Estimate
     GPU utilization using GPU temperature and memory usage. UNIX only. */
  void select_least_utilized_GPU() 
  {
      // Get the number of GPUs on this machine
      int num_devices;
      gpuErrchk( cudaGetDeviceCount(&num_devices) );
      if(num_devices == 0) {
          printf("select_coldest_GPU: Error - No GPU detected\n");
          return;
      }
      // Read GPU "nvidia-smi" info into buffer "output" fo size MAX_BYTES
      const unsigned int MAX_BYTES = 10000;
      char output[MAX_BYTES];
      FILE *fp = popen("nvidia-smi &> /dev/null", "r");
      size_t bytes_read = fread(output, sizeof(char), MAX_BYTES, fp);
      pclose(fp);
      if(bytes_read == 0) {
          printf("Error - No Temperature could be read\n");
          return;
	    }
      // array to hold GPU temperatures, memory utilizations, and final "score"
      float * temperatures = new float[num_devices]; // Celsius
      float * current_memory_useages = new float[num_devices]; // GB
      float * total_memory_capacities = new float[num_devices]; // GB
      float * memory_ratios = new float[num_devices]; // [0, 1]
      float * utilization_scores = new float[num_devices]; // [0, 2]

      // parse output for temperatures using knowledge of "nvidia-smi" output format
      int itr = 0;
      unsigned int num_temps_parsed = 0;
      while(output[itr] != '\0') {
          if(output[itr] == '%') {
              unsigned int temp_begin = itr + 1;
              while(output[itr] != 'C') {
                  ++itr;
              }
              unsigned int temp_end = itr;
              char this_temperature[32];
              // Read in the characters cooresponding to this temperature
              for(unsigned int j = 0; j < temp_end - temp_begin; ++j) {
                  this_temperature[j] = output[temp_begin + j];
              }
              this_temperature[temp_end - temp_begin + 1] = '\0';
              // Convert string representation to float
              temperatures[num_temps_parsed] = atoi(this_temperature);
              num_temps_parsed++;
          }
          ++itr;
      }
      // Identify the highest temperature for normalization and identify 
      // the lowest temperature for printing to terminal.
      float max_temp = temperatures[0];
      float min_temp = temperatures[0];
      for (int i = 1; i < num_devices; i++) 
      {
          float candidate_max = temperatures[i];
          if(candidate_max > max_temp) 
          {
              max_temp = candidate_max;
          }
          float candidate_min = temperatures[i];
          if(candidate_min < min_temp)
          {
              min_temp = candidate_min;
          }
      }
      // Normalize temepratures for scoring
      for (int i = 0; i < num_devices; i++) 
      {
          temperatures[i] /= max_temp;
      }

      // iterate through GPUs for get their memory utilization defined as:
      // free_bytes / total_memory_capacity
      // Also store free_bytes, and total_memory_capacities seperately
      // for printing purposes
      size_t free_memory, total_memory;
      for(int i = 0; i < num_devices; ++i) {
          // Select GPU
          gpuErrchk( cudaSetDevice(i) );
          // Read in available memory on this GPU
          gpuErrchk( cudaMemGetInfo(&free_memory, &total_memory) );
          // Write utilization to array
          total_memory_capacities[i] = (double)total_memory * GB_PER_B;
          current_memory_useages[i] = (double)(total_memory - free_memory) * GB_PER_B;
          memory_ratios[i] = current_memory_useages[i] / total_memory_capacities[i];
          if(DEBUGG == 1)
              printf("total: %f, used: %f, ratio: %f\n", total_memory_capacities[i], current_memory_useages[i], memory_ratios[i]);
      }

      // Assign an overall utilization score to each GPU as:
      // score = noramlized temperature + memory utililization
      for(int i = 0; i < num_devices; ++i) {
          // Read in available memory on this GPU
          utilization_scores[i] = temperatures[i] + memory_ratios[i];
          if(DEBUGG == 1)
            printf("us: %f temp: %f mem: %f\n", utilization_scores[i], temperatures[i], memory_ratios[i]);
      }
      // Identify lowest utilized GPU
      float min_score = utilization_scores[0];
      int index_of_min = 0;
      for(int i = 1; i < num_devices; ++i) {
          float candidate_min = utilization_scores[i];
          if (candidate_min < min_score) {
              min_score = candidate_min;
              index_of_min = i;
          }
      }

      // Tell CUDA to use the GPU with the lowest utilization
      unsigned int percent_mem = (unsigned int) (memory_ratios[index_of_min] * 100);
      printf("GPU # %d selected. Temperature: %d C. Memory being used: %f GB of %f GB  (%u %%)\n", 
          index_of_min, (int)min_temp, current_memory_useages[index_of_min], 
          total_memory_capacities[index_of_min], percent_mem);
      gpuErrchk( cudaSetDevice(index_of_min) );

      // Free memory and return
      delete [] temperatures;
      delete [] memory_ratios;
      delete [] current_memory_useages;
      delete [] total_memory_capacities;
      delete [] utilization_scores;
      return;
  } // end "void select_least_utilized_GPU()""

  /* Create a child thread that will kill the parent thread after the
     specified time limit has been exceeded */
  void enforce_time_limit(int time_limit) {
      printf("Time limit for this program set to %d seconds\n", time_limit);
      int parent_id = getpid();
      pid_t child_id = fork();
      // The fork call creates a lignering child thread that will 
      // kill the parent thread after the time limit has exceeded
      // If it hasn't already terminated.
      if(child_id == 0) // "I am the child thread"
      {
          sleep(time_limit);
          if( kill(parent_id, SIGTERM) == 0) {
              printf("enforce_time_limit.c: Program terminated"
               " for taking longer than %d seconds\n", time_limit);
          }
          // Ensure that parent was actually terminated
          sleep(2);
          if( kill(parent_id, SIGKILL) == 0) {
              printf("enforce_time_limit.c: Program terminated"
               " for taking longer than %d seconds\n", time_limit);
          } 
          // Child thread has done its job. Terminate now.
          exit(0);
      }
      else // "I am the parent thread"
      {
          // Allow the parent thread to continue doing what it was doing
          return;
      }
  } // end "void enforce_time_limit(int time_limit)


} // end "namespace TA_Utilities"

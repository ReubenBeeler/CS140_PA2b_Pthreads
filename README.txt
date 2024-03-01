Last name of Student 1: Beeler
First name of Student 1: Reuben
Email of Student 1: reubenfbeeler@ucsb.edu
Last name of Student 2: Zolyomi
First name of Student 2: Andras
Email of Student 2: azolyomi@ucsb.edu

Q2.a List parallel code that uses at most two barrier calls inside the while loop

int stop = 0;
char * sync_on_barrier() {
  char * err_msg = "pthread_barrier_wait failed\n";
  int ret = pthread_barrier_wait(&mybarrier);
  // sprintf(&err_msg, "pthread_barrier_wait failed; ret = %i\n", ret); // only used if actually is error
  mu_assert(err_msg, ret == PTHREAD_BARRIER_SERIAL_THREAD || ret == 0);
  return NULL;
}

void work_block(long my_rank) {
  if (my_rank == 0) stop = 0;
  char * err_msg;
  for (int iter = 0; iter < no_iterations; ++iter) {
    int blocksize = matrix_dim/thread_count + (matrix_dim % thread_count == 0 ? 0 : 1);
    int upper_bound = MIN(matrix_dim, ((int)my_rank + 1)*blocksize);
    for (int i = (int)my_rank*blocksize; i < upper_bound; ++i) {
      mv_compute(i);
    }
    err_msg = sync_on_barrier(); if (err_msg) {puts(err_msg); memset(vector_y, 0x69, matrix_dim * sizeof(double)); return;} // ERROR -- what to do?
    
    if (my_rank == 0) {
      // check if error is below threshold
      int i;
      for (i = 0; i < matrix_dim; ++i) {
        if (fabs(vector_x[i] - vector_y[i]) > ERROR_THRESHOLD) {
          break;
        }
      }
      if (i == matrix_dim) stop = 1; // sufficiently low error

      // choose one thread to copy x=y
      memcpy((void *) vector_x, (void *) vector_y, matrix_dim * sizeof(double));
    }
    
    err_msg = sync_on_barrier(); if (err_msg) {puts(err_msg); memset(vector_y, 0x69, matrix_dim * sizeof(double)); return;} // ERROR -- what to do?

    if (stop) break;
  }
} 

void work_blockcyclic(long my_rank) {
  if (my_rank == 0) stop = 0;
  char * err_msg;
  for (int iter = 0; iter < no_iterations; ++iter) {
    for (int i = (int)my_rank*cyclic_blocksize; i < matrix_dim; i += thread_count*cyclic_blocksize) {
      int upper_bound = MIN(matrix_dim, i + cyclic_blocksize);
      for (int j = i; j < upper_bound; ++j) {
        mv_compute(j);
      }
    }
    err_msg = sync_on_barrier(); if (err_msg) {puts(err_msg); memset(vector_y, 0x69, matrix_dim * sizeof(double)); return;} // ERROR -- what to do?
    
    if (my_rank == 0) {
      // check if error is below threshold
      int i;
      for (i = 0; i < matrix_dim; ++i) {
        if (fabs(vector_x[i] - vector_y[i]) > ERROR_THRESHOLD) {
          break;
        }
      }
      if (i == matrix_dim) stop = 1; // sufficiently low error

      // choose one thread to copy x=y
      memcpy((void *) vector_x, (void *) vector_y, matrix_dim * sizeof(double));
    }
    
    err_msg = sync_on_barrier(); if (err_msg) {puts(err_msg); memset(vector_y, 0x69, matrix_dim * sizeof(double)); return;} // ERROR -- what to do?

    if (stop) break;
  }
}

Q2.b Report parallel time, speedup, and efficiency for  the upper triangular test matrix case when n=4096 and t=1024. 
Use 2 threads and 4  threads (1 thread per core) under blocking mapping, and block cyclic mapping with block size 1 and block size 16.    


Parallel Time:

$ # 1 THREAD / CORE (NOT PARALLEL)
$ uptime; ./itmv_mult_test_pth 1; uptime
 00:30:14 up 13:30,  0 users,  load average: 0.42, 0.25, 0.26
Test 12: n=4K t=1K upper block mapping: Wall clock time = 0.094567 with 1 threads
Test 13: n=4K t=1K upper block cylic (r=1): Wall clock time = 0.094633 with 1 threads
Test 14: n=4K t=1K upper block cyclic(r=16): Wall clock time = 0.097900 with 1 threads
Summary: Failed 0 out of 3 tests
 00:30:15 up 13:30,  0 users,  load average: 0.42, 0.25, 0.26

$ # 2 THREADS/CORES
$ uptime; ./itmv_mult_test_pth 2; uptime
 00:27:31 up 13:27,  0 users,  load average: 0.37, 0.18, 0.25
Test 12: n=4K t=1K upper block mapping: Wall clock time = 0.078684 with 2 threads
Test 13: n=4K t=1K upper block cylic (r=1): Wall clock time = 0.049777 with 2 threads
Test 14: n=4K t=1K upper block cyclic(r=16): Wall clock time = 0.052505 with 2 threads
Summary: Failed 0 out of 3 tests
 00:27:31 up 13:27,  0 users,  load average: 0.37, 0.18, 0.25

$ # 4 THREADS/CORES
$ uptime; ./itmv_mult_test_pth 4; uptime
 00:27:34 up 13:27,  0 users,  load average: 0.34, 0.18, 0.25
Test 12: n=4K t=1K upper block mapping: Wall clock time = 0.042851 with 4 threads
Test 13: n=4K t=1K upper block cylic (r=1): Wall clock time = 0.032063 with 4 threads
Test 14: n=4K t=1K upper block cyclic(r=16): Wall clock time = 0.026402 with 4 threads
Summary: Failed 0 out of 3 tests
 00:27:35 up 13:27,  0 users,  load average: 0.34, 0.18, 0.25


2-THREAD STATISTICS:

Method                  speedup     efficiency
--------------------- | -------- | ----------- |
Block:                  1.20185     0.60093
Block-Cyclic (r=1):     1.90114     0.95056
Block-Cyclic (r=16):    1.86458     0.93229


4-THREAD STATISTICS:

Method                  speedup     efficiency
--------------------- | -------- | ----------- |
Block:                  2.20687     0.55171
Block-Cyclic (r=1):     2.95147     0.73786
Block-Cyclic (r=16):    3.70805     0.92701


As you can see from the above statistics, block mapping performed the least well with parallelized upper triangular computations.
As you increase the number of threads, Block-Cyclic (r=16) continuously performs at or above the 90% efficiency threshold, though
Block-Cyclic in general performs better than Block. That is because the cylic distribution takes advantage of the natural structure
of upper triangular matrices, which is that different rows have different runtime complexities (due to different number of nontrivial 
columns in matrix mult) hence performing much better on them.


Please indicate if your evaluation is done on CSIL and if yes, list the uptime index of that CSIL machine.  
- We used CSIL! The uptime was about 0.25 when testing. More uptime information before and after test cases can be seen in file `output.txt`.
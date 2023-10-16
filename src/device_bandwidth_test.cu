
#include <stdio.h>
#include <iostream>
#include <memory>
#include <algorithm>

#include "cuda_utils.h"


// Note: how to use the kernel attributes in a C++-like code?
__global__ void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) y[i] = a*x[i] + y[i];
}

// Note: what is the impact of the constant a? 
__global__ void mul(int n, float *x, float *y)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] + y[i];
}

__global__ void noop(int n, float a, float *x, float *y)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    y[i] ++;
}




void bandwidth_test()
{
  const int N = 1 << 20;


  std::unique_ptr<float[]> x = std::make_unique<float[]>(N);
  std::unique_ptr<float[]> y = std::make_unique<float[]>(N);
  auto d_x = cuda::mem_allocate<float>(N);
  auto d_y = cuda::mem_allocate<float>(N);

  std::fill_n(x.get(), N, 1.0f);
  std::fill_n(y.get(), N, 2.0f);

  cuda::copy_to_device(d_x.get(), x.get(), N);
  cuda::copy_to_device(d_y.get(), y.get(), N);

  bool use_noop = false;
  auto f = !use_noop ? saxpy : noop;
  auto milliseconds = cuda::time_it_n(10, [&]{

      f<<<(N+255)/256, 256>>>(N, 2.0f, d_x.get(), d_y.get());
      //mul<<<(N+255)/256, 256>>>(N, d_x.get(), d_y.get());
      //std::cout << "do some work \n"; 
  }
  );

  cuda::copy_to_host(y.get(), d_y.get(), N);

  int n_operations = use_noop ? 2 : 3;
  int n_bytes = sizeof(float);
  std::cout << "Effective Bandwidth (GB/s): " << N*n_bytes*n_operations/milliseconds/1e6 << '\n';
  std::cout << "Time elapsed in milliseconds: " << milliseconds << '\n';

}

int main()
{
  bandwidth_test(); 
}

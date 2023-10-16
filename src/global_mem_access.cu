#include <stdio.h>
#include <iostream>
#include <memory>
#include <algorithm>

#include "cuda_utils.h"

__global__ void access_offset(float* d, int offset)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x + offset; 
  d[i] ++; 
}

__global__ void access_stride(float* d, int s)
{
  int i = (blockDim.x * blockIdx.x + threadIdx.x) * s;
  d[i] ++;
}

void access_offset_experiments(std::ostream& os = std::cout)
{
    int block_size{256}; 
    int mb = 4;
    int n = mb * 1024 * 1024 / sizeof(float);
    auto d = cuda::mem_allocate<float>(33 * n);  

    access_offset<<<n/block_size, block_size>>>(d.get(), 0);   
    os << "offset access test: \n"; 
    for (int i{0}; i <= 32; i++)
    {
      auto ms = cuda::time_it(1, 
        [&](){  access_offset<<<n/block_size, block_size>>>(d.get(), i);}
      ); 
      os << i << ',' << ms << ',' << 2* mb/ms << '\n';   // c MB/ms  = (c/1024)*1000  GB/s
    } 

    os << "\n\n"; 

    access_stride<<<n/block_size, block_size>>>(d.get(), 0);   
    os << "stride access test: \n"; 
    for (int i{0}; i <= 32; i++)
    {
      auto ms = cuda::time_it(1, 
        [&](){  access_stride<<<n/block_size, block_size>>>(d.get(), i);}
      ); 
      os << i << ',' << ms << ',' << 2* mb/ms << '\n';   // c MB/ms  = (c/1024)*1000  GB/s
    } 
}

void run_experiments(std::ostream& os = std::cout) 
{
  access_offset_experiments();
}

int main() 
{
  run_experiments(); 
}
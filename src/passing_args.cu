#include <stdio.h>
#include <iostream>
#include <memory>
#include <algorithm>
#include <cassert>
#include <thread>
#include <chrono>

#include "cuda_utils.h"

const int f1_a{42};   // Ok. 
// int f1_a = 42;  //a host variable "f1_a" cannot be directly read in a device function
__global__ void f1()
{
  printf("Hi world f1: %d \n", f1_a);
}

// int f2_a_v = 10;   // identifier "f2_a_v" is undefined in device code
const int f2_a_v = 10;    // Ok.
const int* const f2_a = &f2_a_v;
__global__ void f2() 
{
  printf("Hi world f2: %d \n", *f2_a);
}

__managed__ int f3_a = 501; 
//__managed__ int* f3_b = &f3_a; 
__global__ void f3() 
{
  printf("Hi world f3: %d \n", f3_a);
}

__constant__ int f4_a;  
__global__ void f4();
void test_f4() 
{
  int f4_h = 54; 
  //cudaMemcpyToSymbol(f4_a, &factor, sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(f4_a, &f4_h, sizeof(int));
  f4<<<1,1>>>();
  cudaDeviceSynchronize();
  
  f4_h = 45; 
  //cudaMemcpyToSymbol(f4_a, &factor, sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(f4_a, &f4_h, sizeof(int));
  f4<<<1,1>>>();
}

__global__ void f4()
{
  printf("Hi world f4: %d \n", f4_a);
}

int main() 
{
  f1<<<1,1>>>(); 
  f2<<<1,1>>>();
  f3<<<1,1>>>();
  test_f4(); 
  cudaDeviceSynchronize();

  
}
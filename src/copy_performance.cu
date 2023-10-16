#include <stdio.h>
#include <iostream>
#include <memory>
#include <algorithm>
#include <cassert>
#include <thread>
#include <chrono>

#include "cuda_utils.h"


struct Params 
{
  int ne = 2; 
  int nt = 4; 
  int nb = 256; 
  int nx = 2 * 1024;
  int ny = 2 * 1024; 
  int n = nx * ny; 
}; 
const Params p{}; 
Params const* const pp{nullptr}; 
const int ne = p.ne;  
const int nt = p.nt; 
const int nb = p.nb;
const int nx = p.nx; 
const int ny = p.ny; 
const int n = nx * ny; 




namespace v1
{
  
  __global__ void copy(float* odata, const float* idata)
  {

    const auto bx = blockIdx.x; 
    const auto by = blockIdx.y; 
    const auto tx = threadIdx.x; 
    const auto ty = threadIdx.y;

    int se = pp->nx / pp->ne; 
    int sb = se / p.nb; 
    int st = sb / p.nt; 
    assert( st == 1);  

    for (int ex = 0; ex < p.ne; ex ++) 
    {
      for (int ey = 0; ey < p.ne; ey ++) 
      {
        int xe = se * ex + sb * bx + st * tx; 
        int ye = se * ey + sb * by + st * ty; 
        int xy_e = xe * nx + ye; 
        odata[xy_e] = idata[xy_e];
      }
    }
  }
} // namespace v1


namespace v2
{
  
  __global__ void copy(float* odata, const float* idata)
  { 
    const auto bx = blockIdx.x; 
    const auto by = blockIdx.y; 
    const auto tx = threadIdx.x; 
    const auto ty = threadIdx.y;

    int sb = nx / nb; 
    int st = sb / nt; 
    int se = st / ne; 
    assert(se == 1);  

    for (int ex = 0; ex < ne; ex ++) 
    {
      for (int ey = 0; ey < ne; ey ++) 
      {
        int xe = se * ex + sb * bx + st * tx; 
        int ye = se * ey + sb * by + st * ty; 
        int xy_e = xe * nx + ye; 
        odata[xy_e] = idata[xy_e];
      }
    }
  }
} // namespace v2

namespace v3
{
  
  __global__ void copy(float* odata, const float* idata)
  {
     
    const auto bx = blockIdx.x; 
    const auto by = blockIdx.y; 
    const auto tx = threadIdx.x; 
    const auto ty = threadIdx.y;

    // version 3: 
    int st = nx / nt; 
    int sb = st / nb; 
    int se = sb / ne; 
    assert(se == 1);   

    for (int ex = 0; ex < ne; ex ++) 
    {
      for (int ey = 0; ey < ne; ey ++) 
      {
        int xe = se * ex + sb * bx + st * tx; 
        int ye = se * ey + sb * by + st * ty; 
        int xy_e = xe * nx + ye; 

        odata[xy_e] = idata[xy_e];
      }
    }
  }
} // namespace v3 




// Check errors and print GB/s
void postprocess(const float *ref, const float *res, int n, float ms)
{
  for (int i = 0; i < n; i++)
  {
    if (res[i] != ref[i]) {
      printf("index : %d, res=%f != ref=%f \n", i, res[i], ref[i]); 
      //std::cout << "index: " << i << " " << res[i] << " " << ref[i] << '\n';
      //std::cout << "*** Failed ***\n"; 

      return;
    }
  }
  std::cout << "Bandwidth: " << 2 * n * sizeof(float) * 1e-6  / ms << " GB/s " 
            << "ms: " << ms << std::endl; 
}

void test() 
{
  /*
  1. Allocate and initiate arrays. 
  */  
  auto h_idata = cuda::mem_allocate_host<float>(n);
  auto h_cdata = cuda::mem_allocate_host<float>(n); 

  auto d_idata = cuda::mem_allocate<float>(n);
  auto d_cdata = cuda::mem_allocate<float>(n);

  dim3 dimGrid(nb, nb, 1);
  dim3 dimBlock(nt, nt, 1);
  
  // host
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx; i++)
      h_idata.get()[j*nx + i] = j*nx + i;
    
  cuda::copy_to_device(d_idata.get(), h_idata.get(), n);

  // warm-up
  //auto copy = v1::copy; 
  int iteration{0}; 
  for (auto copy : {v1::copy, v2::copy, v3::copy})
  {
    v1::copy<<<dimGrid, dimBlock>>>(d_cdata.get(), d_idata.get());

    auto ms = cuda::time_it(100, [&]()
      { v1::copy<<<dimGrid, dimBlock>>>(d_cdata.get(), d_idata.get());}
    ); 
    cuda::copy_to_host(h_cdata.get(), d_cdata.get(), n);
    std::cout << "At iteration " << ++iteration << std::endl; 
    postprocess(h_idata.get(), h_cdata.get(), n, ms);
    //std::this_thread::sleep_for(std::chrono::microseconds{10});
  }


}

int main()
{
  test(); 
}
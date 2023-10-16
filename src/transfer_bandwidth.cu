
#include <stdio.h>
#include <iostream>
#include <memory>
#include <algorithm>

#include "cuda_utils.h"


namespace transfer_bandwidth
{
    namespace v2
    {
      void profile_copies(float *h_a, float *h_b, float *d, unsigned int n, const char* desc)
      {


        auto cp_to_device_time = cuda::time_it(10, 
          [&]() { cuda::copy_to_device(d, h_a, n); }
        ); 
        std::cout << "  " <<  desc << "(v2) Host to Device bandwidth (GB/s): " << (n * sizeof(float) * 1e-6 / cp_to_device_time) << '\n'; 
        
        auto cp_to_host_time = cuda::time_it(10, 
          [&]() { cuda::copy_to_host(h_b, d, n); }
        ); 
        std::cout << "  " << desc << "(v2) Device to Host bandwidth (GB/s): " << (n * sizeof(float) * 1e-6 / cp_to_host_time) << '\n'; 
    


        for (int i = 0; i < n; ++i) {
          if (h_a[i] != h_b[i]) {
            printf("*** %s transfers failed ***\n", desc);
          break;
        }
      }
      }
    }  



    
   
    void test() 
    {
        unsigned int n_elements = 4*1024*1024;
        auto n_bytes = n_elements * sizeof(float); 

        auto h_a = std::make_unique<float[]>(n_elements);
        auto h_b = std::make_unique<float[]>(n_elements); 
        auto d_a = cuda::mem_allocate<float>(n_elements);


        for (int i = 0; i < n_elements; ++i) h_a[i] = i; 

        auto h_a_pinned = cuda::mem_allocate_host<float>(n_elements);
        auto h_b_pinned = cuda::mem_allocate_host<float>(n_elements);
        std::copy_n(h_a_pinned.get(), n_elements, h_a.get()); 
        std::copy_n(h_b_pinned.get(), n_elements, h_b.get());

        // output device info and transfer size
        cudaDeviceProp prop;
        checkCuda( cudaGetDeviceProperties(&prop, 0) );
        std::cout << "\nDevice: " << prop.name << '\n'; 
        std::cout << "Transfer size (MB): " << n_bytes / (1024 * 1024) << std::endl; 

        
        v2::profile_copies(h_a.get(), h_b.get(), d_a.get(), n_elements, "Pageable");
        v2::profile_copies(h_a_pinned.get(), h_b_pinned.get(), d_a.get(), n_elements, "Pinned");
    }
    

}

int main(void)
{
  transfer_bandwidth::test();
}



// See ver1::profile_copies below: 
/*
namespave v1 {
 void profile_copies(float        *h_a, 
                      float        *h_b, 
                      float        *d, 
                      unsigned int  n,
                      const char         *desc)
    {
      printf("\n%s transfers\n", desc);

      unsigned int bytes = n * sizeof(float);

      // events for timing
      cudaEvent_t startEvent, stopEvent; 

      checkCuda( cudaEventCreate(&startEvent) );
      checkCuda( cudaEventCreate(&stopEvent) );

      checkCuda( cudaEventRecord(startEvent, 0) );
      checkCuda( cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice) );
      checkCuda( cudaEventRecord(stopEvent, 0) );
      checkCuda( cudaEventSynchronize(stopEvent) );

      float time;
      checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
      printf("  (v1) Host to Device bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

      checkCuda( cudaEventRecord(startEvent, 0) );
      checkCuda( cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost) );
      checkCuda( cudaEventRecord(stopEvent, 0) );
      checkCuda( cudaEventSynchronize(stopEvent) );

      checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
      printf("  (v1) Device to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

      for (int i = 0; i < n; ++i) {
        if (h_a[i] != h_b[i]) {
          printf("*** %s transfers failed ***\n", desc);
          break;
        }
      }

      // clean up events
      checkCuda( cudaEventDestroy(startEvent) );
      checkCuda( cudaEventDestroy(stopEvent) );
    }    
    } // namespace v1


*/ 
#pragma once 
#include <vector>
// Macros. 

namespace details 
{
inline cudaError_t checkCuda(const char* arg, int line, cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "Error at %i: %s, CUDA Runtime Error: %s\n", line, arg,
            cudaGetErrorString(result));
    //assert(result == cudaSuccess);
  }
  return result; 
}
}

#define checkCuda(res) details::checkCuda(#res, __LINE__, res); 




namespace cuda
{

    template <typename T>
    struct CudaMallocDeleter
    {
      void operator()(T* ptr)
      {
          //std::cout << "Freeing cuda malloc " << std::endl;
          cudaFree(ptr);
      }
    };

    template <typename T>
    std::unique_ptr<T, CudaMallocDeleter<T>> mem_allocate(int n)
    {
        T* ptr{nullptr};
        cudaError_t status = cudaMalloc(&ptr, n * sizeof(T));
        if (status != cudaSuccess)
        {
            throw std::runtime_error{"Failed to allocate host memory"};
        }
        return std::unique_ptr<T, CudaMallocDeleter<T>>{ptr};
    }

    template <typename T>
    struct CudaMallocHostDeleter
    {
      void operator()(T* ptr)
      {
          //std::cout << "Freeing cuda malloc " << std::endl;
          cudaFreeHost(ptr);
      }
    };

    template <typename T> 
    std::unique_ptr<T, CudaMallocHostDeleter<T>> mem_allocate_host(int n)
    {
        T* ptr{nullptr}; 
        cudaError_t status = cudaMallocHost(&ptr, n * sizeof(T)); 
        if (status != cudaSuccess)
        {
          throw std::runtime_error{"Failed to allocate pinned memory"}; 
        }
        return std::unique_ptr<T, CudaMallocHostDeleter<T>>{ptr}; 
    }

    template <typename T>
    void copy_to_device(T* dest, T* src, int n)
    {
        cudaMemcpy(dest, src, n * sizeof(float), cudaMemcpyHostToDevice);
    }

    template <typename T>
    void copy_to_host(T* dest, T* src, int n)
    {
        cudaMemcpy(dest, src, n * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // TODO: events should be destroyed. 
    /*
    struct Event
    {
      Event() 
      {
        cudaEventCreate(&_event);
      }
      ~Event() 
      {
        cudaEventDestroy(_event);
      }
      void record() 
      {
        cudaEventRecord(event);
      }

      CudaEvent_t _event; 
    }; 
    */ 
    auto recorded_event()
    {
      cudaEvent_t event;
      cudaEventCreate(&event);
      cudaEventRecord(event, 0);
      return event;
    }

    // TODO: what can be wrong in this pattern?
    template <typename F>
    float time_it(F&& func)
    {
        auto start = cuda::recorded_event();
        std::forward<F&&>(func)();
        auto stop = cuda::recorded_event();

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        return milliseconds;
    }

    template <typename Iterator> 
    double mean(Iterator first, Iterator last)
    {
        double res{};
        double n = 0; 
        for ( ; first != last ;  first ++)
        {
            res += *first; 
            n += 1; 
        } 
        return res/n; 
    }

    template <typename F> 
    double time_it(int n_iterations, F&& func)
    {
        std::vector<float> vec{}; 
        for (int i = 0; i < n_iterations; i ++)
        {
            //std::cout << "time_it : " << i << std::endl;
            auto it_time = time_it(std::forward<F&&>(func)); 
            vec.push_back(it_time); 
        }
        return mean(vec.begin(), vec.end()); 
    }


}


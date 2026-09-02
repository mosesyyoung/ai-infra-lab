#include <cstdio>
#include <cuda_runtime.h>

int main()
{
    int device_count = 0;
    int driver_version = 0;
    int runtime_version = 0;

    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDriverGetVersion(&driver_version);
    cudaRuntimeGetVersion(&runtime_version);

    printf("CUDA devices: %d\n", device_count);

    printf("Driver API version: %d.%d\n",
           driver_version / 1000,
           (driver_version % 1000) / 10);

    printf("Runtime version: %d.%d\n",
           runtime_version / 1000,
           (runtime_version % 1000) / 10);

    if (device_count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        printf("GPU: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n",
               prop.major,
               prop.minor);
    }

    return 0;
}

/**
 * @file query_device.cu
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */

 #include "../common/book.h"

 int main ( void ) {
    cudaDeviceProp prop;

    int count; 
    HANDLE_ERROR( cudaGetDeviceCount( &count ));
    for(int i=0; i < count; i++) {
        HANDLE_ERROR( cudaGetDeviceProperties(&prop, i) );

        printf("  --- General Information for device %d ---\n", i);
        printf("Name: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Clock rate: %d\n", prop.clockRate);

        printf("Device copy overlap:  ");
        if (prop.deviceOverlap) {
            printf("Enable\n");
        } else {
            printf("Disable\n");
        }

        printf("Kernel exection timeout :   ");
        if (prop.kernelExecTimeoutEnabled) {
            printf("Enable\n");
        } else {
            printf("Disable\n");
        }

        printf("   --- Memory Information for device %d --- \n", i);
        printf("Total global mem: %ld\n", prop.totalGlobalMem);
        printf("Total constant Mem: %ld\n", prop.totalConstMem);
        printf("Max mem pitch: %ld\n", prop.memPitch);
        printf("Texture Alignment: %ld\n", prop.textureAlignment);

        printf("   --- MP Information for device %d --- \n", i);
        printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
        printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
        printf("Registers per mp: %d\n", prop.regsPerBlock);
        printf("Threads in wrap: %d\n", prop.warpSize);
        printf("Max Threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max Thread dimensions:  (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions:  (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\n");
    }
 }
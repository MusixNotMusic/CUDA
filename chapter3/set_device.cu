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
    int dev; 
    
    HANDLE_ERROR( cudaGetDeviceCount( &dev ));
    printf("ID of current CUDA device: %d\n", dev);

    memset(&prop, 0, sizeof(cudaDeviceProp));

    prop.major = 1;
    prop.minor = 3;

    HANDLE_ERROR( cudaChooseDevice(&dev, &prop ));
    printf("ID of CUDA device closest to revision 1.3: %d\n", dev);
    HANDLE_ERROR( cudaSetDevice(dev));
 }
#include <hip/hip_runtime.h>
#include <cstdio>
#include "utils.h"
#include "gpufort.h"

__global__ void  heat_eqn(
 double *t_d,
 double *t_old_d,                                                                    
 const int lb1,                                                                            
 const int lb2,                                                                            
 const int ub1,                                                                           
 const int ub2,                                                                           
 double r,                                                                                         
 int ny,                                                                                          
 int nx) {     
 #undef idx
 #define idx(a,b) (a-t_d_lb1+1)+(ub)*(b-lb2)
 int k = 1 + blockIdx.x * blockDim.x + threadIdx.x;
 int j = 1 + blockIdx.y * blockDim.y + threadIdx.y;
 
// int lb1 = t_d_lb1;
// int lb2 = t_d_lb2;
// int s1 = t_d_n1;
// int s2 = t_d_n2;
 if (k >= lb1 && k <= ub1 && j >= lb2 && j <= ub2){
 t_d[idx(j,k)]=(t_old_d[idx(j,k)]+r*(t_old_d[idx((j+1),k)]+t_old_d[idx(j,(k+1))]+t_old_d[idx((j-1),k)]+t_old_d[idx(j,(k-1))]-4*t_old_d[idx(j,k)]));
}  
}

 
extern "C" {

void launch(
 const int sharedmem,
 hipStream_t stream,
 double *t_d,
 double *t_old_d,
 const int lb1,
 const int lb2,
 const int ub1,
 const int ub2,
 double r,
 int ny,
 int nx) {

 #define divideAndRoundUp(x, y) ((x) / (y) + ((x) % (y) != 0))

 const int blockX = 32; 
 const int blockY = 4; 

 dim3 block(blockX,blockY);

 const int nX = ny; 
 const int nY = nx; 

 const int gridX = divideAndRoundUp(nX,blockX);
 const int gridY = divideAndRoundUp(nY,blockY);

 dim3 grid(gridX,gridY);

 hipLaunchKernelGGL((heat_eqn), grid, block, sharedmem, stream, t_d,t_old_d,lb1,lb2,ub1,ub2,r,ny,nx);

}

}  

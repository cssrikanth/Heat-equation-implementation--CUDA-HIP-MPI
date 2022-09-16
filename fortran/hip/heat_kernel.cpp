#include <hip/hip_runtime.h>
#include <cstdio>
#include <float.h>

#ifdef SINGLE_PRECISION
  typedef float real;
#else
  typedef double real;
#endif

//Access the thread index and offset if required
#define __GIDX(idx,off) 1+(threadIdx.idx + blockIdx.idx * blockDim.idx)+((off)-1)

//Loop bounds template
template <typename I, typename E, typename S> __device__ __forceinline__ bool loop_cond(I idx,E end,S stride) {
  return idx <= end;     
  }

//Two loop parallel thread definition
#define TWO_X 128
#define TWO_Y 4

#define divideAndRoundUp(x, y) ((x) / (y) + ((x) % (y) != 0))

//Access 2D arrays with ghost
#define idx(i,j) (((i)-1+ng)+(nx+2*ng)*((j)-1+ng))

//Access 2D arrays for swapping
#define idx_s(i,j) (((i)-1)+ng*((j)-1))

__global__ void  heat_eqn(
 real *Td,
 real *Td_old,                                                                    
 real r,                                                                                         
 int nx,                                                                                          
 int ny,                                                                                          
 int ng) {     
 
 int j = __GIDX(x,1);
 int k = __GIDX(y,1);

 if (loop_cond(j,nx,1) && loop_cond(k,ny,1)){ 
	 Td[idx(j,k)]=(Td_old[idx(j,k)]+r*(Td_old[idx((j+1),k)]+Td_old[idx(j,(k+1))]+Td_old[idx((j-1),k)]+Td_old[idx(j,(k-1))]-4*Td_old[idx(j,k)]));
 }
}

 
extern "C" void launch_heat_eqn(
 real *Td,
 real *Td_old,
 real r,
 int nx,
 int ny,
 int ng) {

 dim3 block(TWO_X,TWO_Y);
 dim3 grid(divideAndRoundUp(nx,block.x),divideAndRoundUp(ny,block.y));

 hipLaunchKernelGGL((heat_eqn),grid,block,0,0,Td,Td_old,r,nx,ny,ng);

}

__global__ void  swap_recv1(
 real *Td,
 real *Tdr,
 int nx,                                                                                          
 int ny,                                                                                          
 int ng) {     
 
 int i = __GIDX(x,1);
 int j = __GIDX(y,1);
 
 if (loop_cond(i,ng,1) && loop_cond(j,ny,1)){ 
	 Td[idx(i-ng,j)]=Tdr[idx_s(i,j)];
 }
}

extern "C" void launch_recv1(
 real *Td,
 real *Tdr,
 int nx,
 int ny,
 int ng) {

 dim3 block(TWO_X,TWO_Y);
 dim3 grid(divideAndRoundUp(ng,block.x),divideAndRoundUp(ny,block.y));
 
 hipLaunchKernelGGL((swap_recv1), grid, block, 0,0, Td,Tdr,nx,ny,ng);

}

__global__ void  swap_recv2(
 real *Td,
 real *Tdr,
 int nx,                                                                                          
 int ny,                                                                                          
 int ng) {     
 int i = __GIDX(x,1);
 int j = __GIDX(y,1);
 
 if (loop_cond(i,ng,1) && loop_cond(j,ny,1)){ 
	 Td[idx(nx+i,j)]=Tdr[idx_s(i,j)];
 }
}

extern "C" void launch_recv2(
 real *Td,
 real *Tdr,
 int nx,
 int ny,
 int ng) {
 
 dim3 block(TWO_X,TWO_Y);
 dim3 grid(divideAndRoundUp(ng,block.x),divideAndRoundUp(ny,block.y));

 hipLaunchKernelGGL((swap_recv2),grid,block,0,0,Td,Tdr,nx,ny,ng);

}

__global__ void  swap_send(
 real *Td,
 real *Tds1,
 real *Tds2,
 int nx,                                                                                          
 int ny,
 int ng) {     
 
 int i = __GIDX(x,1);
 int j = __GIDX(y,1);
 
 if (loop_cond(i,ng,1) && loop_cond(j,ny,1)){ 
	 Tds1[idx_s(i,j)]=Td[idx(i,j)];
 	 Tds2[idx_s(i,j)]=Td[idx(nx-ng+i,j)];
 }
}

extern "C" void launch_send(
 real *Td,
 real *Tds1,
 real *Tds2,
 int nx,
 int ny,
 int ng) {
 
 dim3 block(TWO_X,TWO_Y);
 dim3 grid(divideAndRoundUp(ng,block.x),divideAndRoundUp(ny,block.y));

 hipLaunchKernelGGL((swap_send),grid,block,0,0,Td,Tds1,Tds2,nx,ny,ng);

}

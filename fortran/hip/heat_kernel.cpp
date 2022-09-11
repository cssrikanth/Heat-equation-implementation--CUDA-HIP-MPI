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
 real *t_d,
 real *t_old_d,                                                                    
 real r,                                                                                         
 int ng,                                                                                          
 int ny,                                                                                          
 int nx) {     
 int j = __GIDX(x,1-ng);
 int k = __GIDX(y,1-ng);

 if (loop_cond(j,ny+ng,1) && loop_cond(k,nx+ng,1)){ 
	 t_d[idx(j,k)]=(t_old_d[idx(j,k)]+r*(t_old_d[idx((j+1),k)]+t_old_d[idx(j,(k+1))]+t_old_d[idx((j-1),k)]+t_old_d[idx(j,(k-1))]-4*t_old_d[idx(j,k)]));
 }
}

 
extern "C" void launch_heat_eqn(
 real *t_d,
 real *t_old_d,
 real r,
 int ng,
 int ny,
 int nx) {

 dim3 block(TWO_X,TWO_Y);
 dim3 grid(divideAndRoundUp(ny+ng-(1-ng)+1,block.x),divideAndRoundUp(nx+ng-(1-ng)+1,block.y));

 hipLaunchKernelGGL((heat_eqn),grid,block,0,0,t_d,t_old_d,r,ng,ny,nx);

}

__global__ void  swap_recv1(
 real *t_d,
 real *tdr,
 int ng,                                                                                          
 int ny,                                                                                          
 int nx) {     
 
 int j = __GIDX(x,1);
 int k = __GIDX(y,1);
 
 if (loop_cond(j,ng,1) && loop_cond(k,ny,1)){ 
	 t_d[idx(j-ng,k)]=tdr[idx_s(j,k)];
 }
}

extern "C" void launch_recv1(
 real *t_d,
 real *tdr,
 int ng,
 int ny,
 int nx) {

 dim3 block(TWO_X,TWO_Y);
 dim3 grid(divideAndRoundUp(ng,block.x),divideAndRoundUp(ny,block.y));
 
 hipLaunchKernelGGL((swap_recv1), grid, block, 0,0, t_d,tdr,ng,ny,nx);

}

__global__ void  swap_recv2(
 real *t_d,
 real *tdr,
 int ng,                                                                                          
 int ny,                                                                                          
 int nx) {     
 int j = __GIDX(x,1);
 int k = __GIDX(y,1);
 
 if (loop_cond(j,ng,1) && loop_cond(k,ny,1)){ 
	 t_d[idx(nx+j,k)]=tdr[idx_s(j,k)];
 }
}

extern "C" void launch_recv2(
 real *t_d,
 real *tdr,
 int ng,
 int ny,
 int nx) {
 
 dim3 block(TWO_X,TWO_Y);
 dim3 grid(divideAndRoundUp(ng,block.x),divideAndRoundUp(ny,block.y));

 hipLaunchKernelGGL((swap_recv2),grid,block,0,0,t_d,tdr,ng,ny,nx);

}

__global__ void  swap_send(
 real *t_d,
 real *tds1,
 real *tds2,
 int ng,                                                                                          
 int ny,
 int nx) {     
 
 int j = __GIDX(x,1);
 int k = __GIDX(y,1);
 
 if (loop_cond(j,ng,1) && loop_cond(k,ny,1)){ 
	 tds1[idx_s(j,k)]=t_d[idx(j,k)];
 	 tds2[idx_s(j,k)]=t_d[idx(nx-ng+j,k)];
 }
}

extern "C" void launch_send(
 real *t_d,
 real *tds1,
 real *tds2,
 int ng,
 int ny,
 int nx) {
 
 dim3 block(TWO_X,TWO_Y);
 dim3 grid(divideAndRoundUp(ng,block.x),divideAndRoundUp(ny,block.y));

 hipLaunchKernelGGL((swap_send),grid,block,0,0,t_d,tds1,tds2,ng,ny,nx);

}

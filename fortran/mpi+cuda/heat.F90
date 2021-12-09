module run
#ifdef USE_CUDA
 use cudafor
#endif
 use mpi

 implicit none

 integer, parameter :: doubtype = selected_real_kind(15,307)  ! double precision
 
 integer, parameter :: dp = doubtype
 integer, parameter :: mpi_prec = mpi_real8

 real(dp), allocatable, dimension(:,:) :: T_d,T_old_d
 real(dp), allocatable, dimension(:,:) :: T
 real(dp), allocatable, dimension(:,:) :: t_1s,t_2s,t_1r,t_2r
 real(dp), allocatable, dimension(:)   :: x,y
 real(dp), allocatable, dimension(:)   :: xg
 
 integer    :: n, ntime
 real(dp)   :: dom_len, delta, dt, nu, r, sigma
 ! If solution file needs to be printed
 integer :: soln
 integer    :: iercuda

 !MPI
 integer, parameter :: ndims = 1
 integer, dimension(:), allocatable  :: nblocks
 logical, dimension(:), allocatable  :: pbc
 integer, dimension(mpi_status_size) :: istatus

 integer, dimension(:), allocatable :: ncoords
 integer :: mp_cart,mp_cartx
 integer :: nrank,nproc,nrank_x
 integer :: ileftx,irightx
 integer :: iermpi

 integer :: nx
 integer :: ny
 integer, parameter :: ng=1   ! Number of ghost nodes
 logical :: masterproc

#ifdef USE_CUDA
 attributes(device) :: T_d, T_old_d
#endif

end module

module mod_setup
 contains
  subroutine setup()
   use run
   implicit none

   integer :: dims(1),i,j,k,ii,jj
   logical :: reord,remain_dims(ndims)
   
   call mpi_init(iermpi)
   call mpi_comm_rank(mpi_comm_world,nrank,iermpi)
   call mpi_comm_size(mpi_comm_world,nproc,iermpi)

   allocate(ncoords(ndims))
   allocate(nblocks(ndims))
   allocate(pbc(ndims))

   pbc(1) = .false.

   masterproc = .false.
   if (nrank==0) masterproc = .true.

   open(unit=11,file='input.dat',form='formatted')

   read (11,*) n, sigma, nu, dom_len, ntime, soln

   close(11)

   dims = [0]
   call MPI_Dims_create(nproc, 1, dims, iermpi)
   nblocks(1) = dims(1)
   if (masterproc) write(*,*)'Automatic MPI decomposition:', nblocks(1),' x 1'

   nx = n/nblocks(1)
   ny = n/1

   reord = .false.

   call mpi_cart_create(mpi_comm_world,ndims,nblocks,pbc,reord,mp_cart,iermpi)
   call mpi_cart_coords(mp_cart,nrank,ndims,ncoords,iermpi)

   remain_dims(1) = .true.
   call mpi_cart_sub(mp_cart,remain_dims,mp_cartx,iermpi)
   call mpi_comm_rank(mp_cartx,nrank_x,iermpi)
   call mpi_cart_shift(mp_cartx,0,1,ileftx,irightx,iermpi)

! Allocate variables
 
   allocate(T(1-ng:nx+ng,1-ng:ny+ng))
   allocate(x(1-ng:nx+ng))
   allocate(y(1-ng:ny+ng))
   allocate(T_d(1-ng:nx+ng,1-ng:ny+ng))
   allocate(T_old_d(1-ng:nx+ng,1-ng:ny+ng))
   
   allocate(t_1s(ng,ny))
   allocate(t_2s(ng,ny))
   allocate(t_1r(ng,ny))
   allocate(t_2r(ng,ny))
   
   allocate(xg(1-ng:n+ng))

   x = 0.0_dp
   y = 0.0_dp
   xg = 0.0_dp
   t_1s = 0.0_dp
   t_2s = 0.0_dp
   t_1r = 0.0_dp
   t_2r = 0.0_dp

   delta = dom_len/real(n-1)

   dt = (sigma * delta**2)/nu

   do i=1-ng,n+ng
    xg(i) = (i-1)*delta
   enddo
   ii = nx*ncoords(1)
   do i=1-ng,nx+ng
    x(i) = xg(ii+i)
   enddo
   y = xg

  end subroutine
end module

module mod_swap
 contains
  subroutine swap()
   use run
   implicit none
   integer :: i,j
   integer :: indx,indy

   indx = ng*ny

   !$cuf kernel do(2) <<<*,*>>>
   do j=1,ny
    do i=1,ng
      t_1s(i,j) = T_d(i,j)
      t_2s(i,j) = T_d(nx-ng+i,j)
    end do
   end do
   !@cuf iercuda=cudaDeviceSynchronize()

   call mpi_sendrecv(t_1s,indx,mpi_prec,ileftx ,1,t_2r,indx,mpi_prec,irightx,1,mp_cartx,istatus,iermpi)
   call mpi_sendrecv(t_2s,indx,mpi_prec,irightx,2,t_1r,indx,mpi_prec,ileftx ,2,mp_cartx,istatus,iermpi)
  
   if (ileftx/=mpi_proc_null) then 
    !$cuf kernel do(2) <<<*,*>>>
    do j=1,ny
     do i=1,ng
      T_d(i-ng,j) = t_1r(i,j)
     end do
    end do
    !@cuf iercuda=cudaDeviceSynchronize()
   endif
   if (irightx/=mpi_proc_null) then 
    !$cuf kernel do(2) <<<*,*>>>
    do j=1,ny
     do i=1,ng
      T_d(nx+i,j) = t_2r(i,j)
     end do
    end do
    !@cuf iercuda=cudaDeviceSynchronize()
   end if
   
  end subroutine

end module

module mod_heat
 contains
  subroutine heat_eqn()
   use run
   use mod_swap
   implicit none
   integer    :: i,j,k

   ! Time loop
   do i=1,ntime
    if(masterproc)write(*,*)"time_it:", i
    T_old_d = T_d
    !$cuf kernel do(2) <<<*,*>>>
    do j=1,nx
     do k=1,ny
       T_d(j,k) = T_old_d(j,k) + r*(T_old_d(j+1,k)+T_old_d(j,k+1)+T_old_d(j-1,k)+T_old_d(j,k-1)-4*T_old_d(j,k))
     end do
    end do
    !@cuf iercuda=cudaDeviceSynchronize()

    ! Ghost update
    call swap() 
   end do

  end subroutine 

end module

program heat
 use run
 use mod_heat
 use mod_setup

 implicit none

 integer ::  i, j, k, z
 real(dp) :: start, finish,lsum,gsum
 character(len=8) :: fmt,x1

 call setup()

! Setting initial and boundary conditions
 T = 2.0_dp
 T_d = 2.0_dp
 T_old_d = 2.0_dp
 if (ileftx==mpi_proc_null) then
   T(1-ng,:) = 1.0_dp
 end if
 if (irightx==mpi_proc_null) then
   T(nx+ng,:) = 1.0_dp
 end if
 T(:,1-ng) = 1.0_dp
 T(:,ny+ng) = 1.0_dp

! cfl 
 r = (nu*dt)/delta**2

 call MPI_BARRIER(mpi_comm_world,iermpi) 
 call cpu_time(start)
!
 ! Host to device
 T_d = T

 call heat_eqn()
!
 ! Back to host
 T = T_d
  
 call MPI_BARRIER(mpi_comm_world,iermpi) 
 call cpu_time(finish)

 lsum = 0.0_dp
 do i=1,nx
  do j=1,ny
   lsum = lsum + T(i,j)
  enddo
 enddo
 call MPI_Reduce(lsum, gsum, 1, MPI_DOUBLE, MPI_SUM, 0, mpi_comm_world, iermpi) 

 if(masterproc)print*,"Sum of Temperature:",gsum

 if (soln == 1) then
  fmt = '(I5.5)'

  write (x1,fmt) nrank 
  open(unit=17,file='soln'//trim(x1)//'.dat',form='formatted')
  do i=1,nx
   do j=1,ny
    write(17,*)x(i),y(j),T(i,j)
   enddo
  enddo
  close(17)
 end if
         
!
 if(masterproc)print*,"simulation completed!!!!"
 if(masterproc)print*,"total time:", finish - start

 deallocate(x,y)
 deallocate(T,T_d,T_old_d)
 deallocate(xg)
 deallocate(ncoords,nblocks,pbc)

 call mpi_finalize(iermpi)

end program

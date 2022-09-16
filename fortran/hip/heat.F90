module run

 use hipfort
 use hipfort_check
 use mpi

 implicit none

 integer, parameter :: doubtype = selected_real_kind(15,307)  ! double precision
 
 integer, parameter :: rkind = doubtype
 integer, parameter :: c_rkind = C_DOUBLE
 integer, parameter :: mpi_prec = mpi_real8
 real(rkind), pointer, dimension(:,:) :: Td,Td_old
 
 real(rkind), allocatable, dimension(:,:) :: T
 real(rkind), allocatable, dimension(:,:) :: T1s,T2s,T1r,T2r

 real(rkind), pointer, dimension(:,:) :: Td1s,Td2s,Td1r,Td2r

 real(rkind), allocatable, dimension(:)   :: x,y
 real(rkind), allocatable, dimension(:)   :: xg
 
 integer    :: n, ntime
 real(rkind)   :: dom_len, delta, dt, nu, sigma
 real(rkind) :: r
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
 integer :: iermpi,local_comm,mydev

 integer :: nx
 integer :: ny
 integer, parameter :: ng=1   ! Number of ghost nodes
 logical :: masterproc
 
 interface
  subroutine launch_heat_eqn(Td,Td_old,r,nx,ny,ng) bind(c)
   import :: c_ptr, c_rkind, c_int
   implicit none

   type(c_ptr),value :: Td
   type(c_ptr),value :: Td_old

   real(c_rkind),value :: r
   integer(c_int),value :: nx
   integer(c_int),value :: ny
   integer(c_int),value :: ng
  end subroutine
  
  subroutine launch_send(Td,Tds1,Tds2,nx,ny,ng) bind(c)
   import :: c_ptr,c_int
   
   implicit none

   type(c_ptr),value :: Td
   type(c_ptr),value :: Tds1
   type(c_ptr),value :: Tds2

   integer(c_int),value :: nx
   integer(c_int),value :: ny
   integer(c_int),value :: ng
  end subroutine
  
  subroutine launch_recv1(Td,tdr,nx,ny,ng) bind(c)
   import :: c_ptr,c_int
   
   implicit none

   type(c_ptr),value :: Td
   type(c_ptr),value :: Tdr

   integer(c_int),value :: nx
   integer(c_int),value :: ny
   integer(c_int),value :: ng
  end subroutine
  
  subroutine launch_recv2(Td,Tdr,nx,ny,ng) bind(c)
   import :: c_ptr,c_int
   
   implicit none

   type(c_ptr),value :: Td
   type(c_ptr),value :: Tdr

   integer(c_int),value :: nx
   integer(c_int),value :: ny
   integer(c_int),value :: ng
  end subroutine
 
 end interface

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

   mydev=0
   call mpi_comm_split_type(mpi_comm_world,mpi_comm_type_shared,0,mpi_info_null,local_comm,iermpi)
   call mpi_comm_rank(local_comm,mydev,iermpi)

   call hipCheck(hipSetDevice(mydev))

   write(*,*) "MPI rank",nrank,"using GPU",mydev

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

! Device variables
   call hipCheck(hipMalloc(Td, (nx + ng) - ((1 - ng)) + 1, (ny + ng) - ((1 - ng)) + 1))
   call hipCheck(hipMalloc(Td_old, (nx + ng) - ((1 - ng)) + 1, (ny + ng) - ((1 - ng)) + 1))
   call hipCheck(hipMalloc(td1s, ng, ny))
   call hipCheck(hipMalloc(td2s, ng, ny))
   call hipCheck(hipMalloc(td1r, ng, ny))
   call hipCheck(hipMalloc(td2r, ng, ny))
   
! Host variables
   allocate(T(1-ng:nx+ng,1-ng:ny+ng))
   allocate(x(1-ng:nx+ng))
   allocate(y(1-ng:ny+ng))
   allocate(t1s(ng,ny))
   allocate(t2s(ng,ny))
   allocate(t1r(ng,ny))
   allocate(t2r(ng,ny))
   allocate(xg(1-ng:n+ng))

   delta = dom_len/real(n-1)

   dt = (sigma * delta**2)/nu

   r = (nu*dt)/delta**2

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
 
   call launch_send(c_loc(Td),c_loc(Td1s),c_loc(Td2s),nx,ny,ng)
   call hipCheck(hipDeviceSynchronize())
   
   call hipCheck(hipMemcpy(T1s, Td1s, hipMemcpyDeviceToHost))
   call hipCheck(hipMemcpy(T2s, Td2s, hipMemcpyDeviceToHost))

   call mpi_sendrecv(T1s,indx,mpi_prec,ileftx,1,T2r,indx,mpi_prec,irightx,1,mp_cartx,istatus,iermpi)
   call mpi_sendrecv(T2s,indx,mpi_prec,irightx,2,T1r,indx,mpi_prec,ileftx,2,mp_cartx,istatus,iermpi)

   call hipCheck(hipMemcpy(Td1r, T1r, hipMemcpyHostToDevice))
   call hipCheck(hipMemcpy(Td2r, T2r, hipMemcpyHostToDevice))
!  
   if (ileftx/=mpi_proc_null) then 
    call launch_recv1(c_loc(Td),c_loc(Td1r),nx,ny,ng)
    call hipCheck(hipDeviceSynchronize())
   endif

   if (irightx/=mpi_proc_null) then 
    call launch_recv2(c_loc(Td),c_loc(Td2r),nx,ny,ng)
    call hipCheck(hipDeviceSynchronize())
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
   
    call hipCheck(hipMemcpy(Td_old, Td, hipMemcpyDeviceToDevice))
 
    call launch_heat_eqn(c_loc(Td),c_loc(Td_old),r,nx,ny,ng)
    call hipCheck(hipDeviceSynchronize())

    ! Ghost update
    call swap() 
   end do

  end subroutine 

end module

program heat
 use mod_heat
 use run
 use mod_setup

 implicit none

 integer ::  i, j, k, z
 real(rkind) :: lsum,gsum
 character(len=8) :: fmt,x1
 real(rkind) :: timing(1:2)

 call setup()

 if(masterproc) print*, "nx: ", nx
 if(masterproc) print*, "ny: ", ny

! Setting initial and boundary conditions
 T = 2.0_rkind
 if (ileftx==mpi_proc_null) then
   T(1-ng,:) = 1.0_rkind
 end if
 if (irightx==mpi_proc_null) then
   T(nx+ng,:) = 1.0_rkind
 end if
 T(:,1-ng) = 1.0_rkind
 T(:,ny+ng) = 1.0_rkind

 call MPI_BARRIER(MPI_COMM_WORLD, iermpi) ; timing(1) = MPI_Wtime()

 ! Host to device
 call hipCheck(hipMemcpy(Td, T, hipMemcpyHostToDevice))

 call heat_eqn()

 ! Back to host
 call hipCheck(hipMemcpy(T, Td, hipMemcpyDeviceToHost))
 call hipCheck(hipDeviceSynchronize())
  
 call MPI_BARRIER(MPI_COMM_WORLD, iermpi) ; timing(2) = MPI_Wtime()

 !Debug
 !lsum = 0.0_rkind
 !do i=1,nx
 ! do j=1,ny
 !  lsum = lsum + T(i,j)
 ! enddo
 !enddo
 !call MPI_Reduce(lsum, gsum, 1, MPI_DOUBLE, MPI_SUM, 0, mpi_comm_world, iermpi) 

 !if(masterproc)print*,"Sum of Temperature:",gsum

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
 if(masterproc)print*,"Average time:", (timing(2)-timing(1))/ntime

 call hipCheck(hipFree(Td))
 call hipCheck(hipFree(Td_old))
 call hipCheck(hipFree(Td1s))
 call hipCheck(hipFree(Td2s))
 call hipCheck(hipFree(Td1r))
 call hipCheck(hipFree(Td2r))
 
 deallocate(T)
 deallocate(T1s,T2s,T1r,T2r)
 deallocate(x,y)
 deallocate(xg)
 deallocate(ncoords,nblocks,pbc)

 call mpi_finalize(iermpi)

end program

module run

 use hipfort
 use hipfort_check
 use mpi

 implicit none

 integer, parameter :: doubtype = selected_real_kind(15,307)  ! double precision
 
 integer, parameter :: mykind = doubtype
 integer, parameter :: c_mykind = C_DOUBLE
 integer, parameter :: mpi_prec = mpi_real8
 real(mykind), pointer, dimension(:,:) :: T_d,T_old_d
 
 real(mykind), allocatable, dimension(:,:) :: T
 real(mykind), allocatable, dimension(:,:) :: t_1s,t_2s,t_1r,t_2r

 real(mykind), pointer, dimension(:,:) :: td_1s,td_2s,td_1r,td_2r

 real(mykind), allocatable, dimension(:)   :: x,y
 real(mykind), allocatable, dimension(:)   :: xg
 
 integer    :: n, ntime
 real(mykind)   :: dom_len, delta, dt, nu, sigma
 real(mykind) :: r
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
  subroutine launch(td1,td2,rr,ng,ny,nx) bind(c)
   import :: c_ptr, c_mykind
   implicit none

   type(c_ptr),value :: td1
   type(c_ptr),value :: td2

   real(c_mykind),value :: rr
   integer,value :: ng
   integer,value :: ny
   integer,value :: nx
  end subroutine
  
  subroutine launchs(td,tds1,tds2,ng,ny,nx) bind(c)
   import :: c_ptr
   
   implicit none

   type(c_ptr),value :: td
   type(c_ptr),value :: tds1
   type(c_ptr),value :: tds2

   integer,value :: nx
   integer,value :: ny
   integer,value :: ng
  end subroutine
  
  subroutine launchr1(td,tdr,ng,ny,nx) bind(c)
   import :: c_ptr
   
   implicit none

   type(c_ptr),value :: td
   type(c_ptr),value :: tdr

   integer,value :: ny
   integer,value :: ng
   integer,value :: nx
  end subroutine
  
  subroutine launchr2(td,tdr,ng,ny,nx) bind(c)
   import :: c_ptr
   
   implicit none

   type(c_ptr),value :: td
   type(c_ptr),value :: tdr

   integer,value :: nx
   integer,value :: ny
   integer,value :: ng
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

! Allocate variables
 
! Device variables
   call hipCheck(hipMalloc(T_d, (nx + ng) - ((1 - ng)) + 1, (ny + ng) - ((1 - ng)) + 1))
   call hipCheck(hipMalloc(T_old_d, (nx + ng) - ((1 - ng)) + 1, (ny + ng) - ((1 - ng)) + 1))
   call hipCheck(hipMalloc(td_1s, ng, ny))
   call hipCheck(hipMalloc(td_2s, ng, ny))
   call hipCheck(hipMalloc(td_1r, ng, ny))
   call hipCheck(hipMalloc(td_2r, ng, ny))
   
! Host variables
   allocate(T(1-ng:nx+ng,1-ng:ny+ng))
   allocate(x(1-ng:nx+ng))
   allocate(y(1-ng:ny+ng))
   allocate(t_1s(ng,ny))
   allocate(t_2s(ng,ny))
   allocate(t_1r(ng,ny))
   allocate(t_2r(ng,ny))
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
   type(dim3) :: grid, tBlock


   indx = ng*ny
 
    call launchs(c_loc(T_d),c_loc(td_1s),c_loc(td_2s),ng,ny,nx)
    call hipCheck(hipMemcpy(td_1s, t_1s, hipMemcpyDeviceToHost))
    call hipCheck(hipMemcpy(td_2s, t_2s, hipMemcpyDeviceToHost))

   call mpi_sendrecv(t_1s,indx,mpi_prec,ileftx ,1,t_2r,indx,mpi_prec,irightx,1,mp_cartx,istatus,iermpi)
   call mpi_sendrecv(t_2s,indx,mpi_prec,irightx,2,t_1r,indx,mpi_prec,ileftx ,2,mp_cartx,istatus,iermpi)

    call hipCheck(hipMemcpy(t_1r, td_1r, hipMemcpyHostToDevice))
    call hipCheck(hipMemcpy(t_2r, td_2r, hipMemcpyHostToDevice))
!  
   if (ileftx/=mpi_proc_null) then 
    call launchr1(c_loc(T_d),c_loc(td_1r),ng,ny,nx)
   endif

   if (irightx/=mpi_proc_null) then 
    call launchr2(c_loc(T_d),c_loc(td_2r),ng,ny,nx)
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
   
    call hipCheck(hipMemcpy(T_old_d, T_d, hipMemcpyDeviceToDevice))
 
    call launch(c_loc(T_d),c_loc(T_old_d),r,ng,ny,nx)

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
 real :: start, finish,lsum,gsum
 character(len=8) :: fmt,x1

 call setup()

! Setting initial and boundary conditions
 T = 2.0
 if (ileftx==mpi_proc_null) then
   T(1-ng,:) = 1.0
 end if
 if (irightx==mpi_proc_null) then
   T(nx+ng,:) = 1.0
 end if
 T(:,1-ng) = 1.0
 T(:,ny+ng) = 1.0

 call MPI_BARRIER(mpi_comm_world,iermpi) 
 call cpu_time(start)

 ! Host to device
 call hipCheck(hipMemcpy(T_d, T, hipMemcpyHostToDevice))

 call heat_eqn()

 ! Back to host
 call hipCheck(hipMemcpy(T, T_d, hipMemcpyDeviceToHost))
  
 call MPI_BARRIER(mpi_comm_world,iermpi) 
 call cpu_time(finish)

 lsum = 0.0
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

 call hipCheck(hipFree(T_d))
 call hipCheck(hipFree(T_old_d))
 call hipCheck(hipFree(td_1s))
 call hipCheck(hipFree(td_2s))
 call hipCheck(hipFree(td_1r))
 call hipCheck(hipFree(td_2r))
 
 deallocate(T)
 deallocate(t_1s,t_2s,t_1r,t_2r)
 deallocate(x,y)
 deallocate(xg)
 deallocate(ncoords,nblocks,pbc)

 call mpi_finalize(iermpi)

end program

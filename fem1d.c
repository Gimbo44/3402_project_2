# include <stdlib.h>
# include <stdio.h>
# include <time.h>
# include <sys/time.h>
# include "mpi.h"

int main ( int argc, char *argv[]  );
void assemble ( double adiag[], double aleft[], double arite[], double f[], 
  double h[], int indx[], int nl, int node[], int nu, int nquad, int nsub, 
  double ul, double ur, double xn[], double xquad[] );
double ff ( double x );


void geometry ( double h[],  int nl, int nsub,
                double xl, double xn[], double xquad[], double xr, int offset , int chunksize);
void geometry_plus (int ibc, int indx[], int *nu, int nsub, int node[]);

void init ( int *ibc, int *nquad, double *ul, double *ur, double *xl, 
  double *xr );
void output ( double f[], int ibc, int indx[], int nsub, int nu, double ul, 
  double ur, double xn[] );
void phi ( int il, double x, double *phii, double *phiix, double xleft, 
  double xrite );
double pp ( double x );
void prsys ( double adiag[], double aleft[], double arite[], double f[], 
  int nu );
double qq ( double x );
void solve ( double adiag[], double aleft[], double arite[], double f[], 
  int nu );
void timestamp ( void );

/******************************************************************************/

int main ( int argc, char *argv[]  ) {
  struct timeval start, end;
  gettimeofday(&start, NULL);
//# define NSUB atoi(argv[1])
# define NSUB atoi(argv[1])
# define NL 20 
# define  MASTER 0
/*
    * ==================================================
    * OPENMPI INITIALIZATION + related variable initialization
    */
  int numtasks;
  int taskid;
  int rc;
  int dest;
  int offset;
  int source;
  int chunksize;
  MPI_Status status;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD,&taskid);

  /*
  * ==================================================
  */
  /*
   * ===================================================================================================================
   *  Project 2 comments
   * ===================================================================================================================
   * Version: 1.0
   * Below is the result of implementing openmpi to the project. A local and master copy of most of the variables
   * were necessary. I tried to be clever by only initializing the master variables inside the master process.
   * ===================================================================================================================
  */
  double *adiag;
  double *aleft;
  double *arite;
  double *f;
  double *h;
  int ibc;
  int *indx;
  int *node;
  int nquad;
  int nu;
  double ul;
  double ur;
  double xl;
  double *xn;
  double *xquad;
  double xr;

  double *local_adiag;
  double *local_aleft;
  double *local_arite;
  double *local_f;
  double *local_h;
  int local_nquad;
  double local_ul;
  double local_ur;
  double local_xl;
  double *local_xn;
  double *local_xquad;
  double local_xr;

    if(taskid == MASTER){
      adiag = (double *) malloc(sizeof(double) * (NSUB + 1));
      //double aleft[NSUB+1];
      aleft = (double *) malloc(sizeof(double) * (NSUB + 1));
      //double arite[NSUB+1];
      arite = (double *) malloc(sizeof(double) * (NSUB + 1));
      //double f[NSUB+1];
      f = (double *) malloc(sizeof(double) * (NSUB + 1));
      //double h[NSUB];
      h = (double *) malloc(sizeof(double) * (NSUB + 1));
      //int indx[NSUB+1];
      indx = (int *) malloc(sizeof(int) * (NSUB + 1));
      //int node[NL*NSUB];
      node = (int *) malloc(sizeof(int) * (NL * NSUB + 1));
      //double xn[NSUB+1];
      xn = (double *) malloc(sizeof(double) * (NSUB + 1));
      //double xquad[NSUB];
      xquad = (double *) malloc(sizeof(double) * (NSUB + 1));
    }

    local_adiag = (double *) malloc(sizeof(double) * (NSUB + 1));
    //double aleft[NSUB+1];
    local_aleft = (double *) malloc(sizeof(double) * (NSUB + 1));
    //double arite[NSUB+1];
    local_arite = (double *) malloc(sizeof(double) * (NSUB + 1));
    //double f[NSUB+1];
    local_f = (double *) malloc(sizeof(double) * (NSUB + 1));
    //double h[NSUB];
    local_h = (double *) malloc(sizeof(double) * (NSUB + 1));
    //double xn[NSUB+1];
    local_xn = (double *) malloc(sizeof(double) * (NSUB + 1));
    //double xquad[NSUB];
    local_xquad = (double *) malloc(sizeof(double) * (NSUB + 1));



  timestamp ( );


  //printf ( "\n" );
  //printf ( "FEM1D\n" );
  //printf ( "  C version\n" );
  //printf ( "\n" );
  //printf ( "  Solve the two-point boundary value problem\n" );
  //printf ( "\n" );
  //printf ( "  - d/dX (P dU/dX) + Q U  =  F\n" );
  //printf ( "\n" );
  //printf ( "  on the interval [XL,XR], specifying\n" );
  //printf ( "  the value of U or U' at each end.\n" );
  //printf ( "\n" );
  //printf ( "  The interval [XL,XR] is broken into NSUB = %d subintervals\n", NSUB );
  //printf ( "  Number of basis functions per element is NL = %d\n", NL );

  /*
   * ===================================================================================================================
   *  Project 2 comments
   * ===================================================================================================================
   * Version: 1.0
   * This version of the project will be focusing on a pure MPI implementation. Doing this allows for a pure vs hybrid comparison.
   * Initially looking at the structure, there appeared to be only three places which could benefit from an MPI implementation.
   * These three places were the geometry, assemble and output function. Prsys showed some potential but will be looked into futher
   * down the road.
   *
   * The first task consisted on focusing all efferts to distribute the work of the geometry function across x computers
   * in our cluster. Looking into the geometry function, it can be seen that six things are occurring. All of which involve
   * variable initialization/assigning values.
   *
   * The first job was breaking up the arrays into sub-arrays. Since all arrays in geometry were built to hold NSUB
   * elements, it made sense to divide NSUB by the number of processes. This would give us the chunk-size of the work
   * being distributed. The second part involved the offset, where the processes should start from. This was simple enough
   * by multiplying the taskid by the chunksize.
   *
   * Once all work was done, MPI_Reduce was used to combine the x arrays into a single global array.
   *
   * Results:
   * What was found was increasing the size of nsub resulted in performance loss. As the number of cores got
   * larger, the run-time increased (got worse). This might be due to the fact the MPI_Reduce has additional
   * computation overhead due to more data sources.
   *
   * ===================================================================================================================
   */

  int i;

/*
  Initialize the data.
*/

  init ( &ibc, &nquad, &ul, &ur, &xl, &xr );

/*
  Compute the geometric quantities.
*/
/*
   * ===================================================================================================================
   *  Project 2 comments
   * ===================================================================================================================
   * Version: 1.0
   * The data arrays were broken up into large sections which will, in the future, be futher optimized by incorporating openmp.
   * Initially, to get the project started, I implemented a poor mechanism to part off any additional data which might have been missed
   * in the case where the NSUB/numtasks didn't produce a nice number. Improvement in distributing this additional data will be
   * achieved in later versions.
   * ===================================================================================================================
  */
  chunksize = (NSUB/numtasks);
  offset = (int)(chunksize*taskid);
  if(taskid != MASTER) {
    if(taskid+1 == numtasks){
      // This is the current code mechanism, if the current process is the last simply make the end put the value of NSUB.
      geometry(local_h, NL, NSUB, xl, local_xn, local_xquad, xr, offset, NSUB);
    }else{
      geometry(local_h, NL, NSUB, xl, local_xn, local_xquad, xr, offset, chunksize + offset);
    }

  }else{
    geometry(local_h, NL, NSUB, xl, local_xn, local_xquad, xr, offset, chunksize);
  }
  ////printf("task %d chunksize = %d offset = %d, range = %d\n",taskid,chunksize,offset, chunksize + offset);
/*
   * ===================================================================================================================
   *  Project 2 comments
   * ===================================================================================================================
   * Version: 1.0
   * Below consists of the three data loops that are contained in geometry that need to be recombined.
   * ===================================================================================================================
   */

  MPI_Reduce(local_xn, xn,NSUB+1,MPI_DOUBLE,MPI_MAX, MASTER, MPI_COMM_WORLD);
  // had to use max for ^^ because there is a necessary overlap.
  MPI_Reduce(local_h, h,NSUB+1,MPI_DOUBLE,MPI_SUM, MASTER, MPI_COMM_WORLD);

  MPI_Reduce(local_xquad, xquad,NSUB+1,MPI_DOUBLE,MPI_SUM, MASTER, MPI_COMM_WORLD);




  if(taskid == MASTER) {
    //Locate the geometry_plus function to read up on the reasoning behind the additional function.
    geometry_plus(ibc,indx,&nu,NSUB,node);

/*
  Assemble the linear system.
*/
    assemble(adiag, aleft, arite, f, h, indx, NL, node, nu, nquad,NSUB, ul, ur, xn, xquad);
/*
  Print out the linear system.
*/
    prsys(adiag, aleft, arite, f, nu);
/*
  Solve the linear system.
*/
    solve(adiag, aleft, arite, f, nu);
/*
  Print out the solution.
*/
    output(f, ibc, indx, NSUB, nu, ul, ur, xn);
/*
  Terminate.
*/
    //printf ( "\n" );
    //printf ( "FEM1D:\n" );
    //printf ( "  Normal end of execution.\n" );

    //printf ( "\n" );
    timestamp();
    gettimeofday(&end, NULL);
    double delta = ((end.tv_sec - start.tv_sec) * 1000000u +
                    end.tv_usec - start.tv_usec) / 1.e6;

    printf("%12.10f\n", delta);
  }
  MPI_Finalize();
  return 0;
# undef NL
# undef NSUB
# undef MASTER
}
/******************************************************************************/

void assemble ( double adiag[], double aleft[], double arite[], double f[], 
  double h[], int indx[], int nl, int node[], int nu, int nquad, int nsub, 
  double ul, double ur, double xn[], double xquad[] )
{
  double aij;
  double he;
  int i;
  int ie;
  int ig;
  int il;
  int iq;
  int iu;
  int jg;
  int jl;
  int ju;
  double phii;
  double phiix;
  double phij;
  double phijx;
  double x;
  double xleft;
  double xquade;
  double xrite;

  for ( i = 0; i < nu; i++ )
  {
    f[i] = 0.0;
  }
  for ( i = 0; i < nu; i++ )
  {
    adiag[i] = 0.0;
  }
  for ( i = 0; i < nu; i++ )
  {
    aleft[i] = 0.0;
  }
  for ( i = 0; i < nu; i++ )
  {
    arite[i] = 0.0;
  }

  for ( ie = 0; ie < nsub; ie++ )
  {
    he = h[ie];
    xleft = xn[node[0+ie*2]];
    xrite = xn[node[1+ie*2]];

    for ( iq = 0; iq < nquad; iq++ )
    {
      xquade = xquad[ie];

      for ( il = 1; il <= nl; il++ )
      {
        ig = node[il-1+ie*2];
        iu = indx[ig] - 1;

        if ( 0 <= iu )
        {
          phi ( il, xquade, &phii, &phiix, xleft, xrite );
          f[iu] = f[iu] + he * ff ( xquade ) * phii;

          if ( ig == 0 )
          {
            x = 0.0;
            f[iu] = f[iu] - pp ( x ) * ul;
          }
          else if ( ig == nsub )
          {
            x = 1.0;
            f[iu] = f[iu] + pp ( x ) * ur;
          }

          for ( jl = 1; jl <= nl; jl++ )
          {
            jg = node[jl-1+ie*2];
            ju = indx[jg] - 1;

            phi ( jl, xquade, &phij, &phijx, xleft, xrite );

            aij = he * ( pp ( xquade ) * phiix * phijx 
                       + qq ( xquade ) * phii  * phij   );

            if ( ju < 0 )
            {
              if ( jg == 0 )
              {
                f[iu] = f[iu] - aij * ul;
              }
              else if ( jg == nsub )
              {               
                f[iu] = f[iu] - aij * ur;
              }
            }

            else
            {
              if ( iu == ju )
              {
                adiag[iu] = adiag[iu] + aij;
              }
              else if ( ju < iu )
              {
                aleft[iu] = aleft[iu] + aij;
              }
              else
              {
                arite[iu] = arite[iu] + aij;
              }
            }
          }
        }
      }
    }
  }
  return;
}
/******************************************************************************/

double ff ( double x )

/******************************************************************************/
{
  double value;

  value = 0.0;

  return value;
}
/******************************************************************************/

void geometry ( double h[],  int nl, int nsub,
                double xl, double xn[], double xquad[], double xr, int offset , int chunksize)

/******************************************************************************/

{
  int i;

  //printf ( "\n" );
  //printf ( "  Node      Location\n" );
  //printf ( "\n" );
  for ( i = offset; i <= chunksize; i++ )
  {
    xn[i]  =  ( ( double ) ( nsub - i ) * xl 
              + ( double )          i   * xr ) 
              / ( double ) ( nsub );
    //printf ( "  %8d  %14f \n", i, xn[i] );
  }

  //printf ( "\n" );
  //printf ( "Subint    Length\n" );
  //printf ( "\n" );

  for ( i = offset; i < chunksize; i++ )
  {
    h[i] = xn[i+1] - xn[i];
    //printf ( "  %8d  %14f\n", i+1, h[i] );
  }

  //printf ( "\n" );
  //printf ( "Subint    Quadrature point\n" );
  //printf ( "\n" );
  for ( i = offset; i < chunksize; i++ )
  {
    xquad[i] = 0.5 * ( xn[i] + xn[i+1] );
    //printf ( "  %8d  %14f\n", i+1, xquad[i] );
  }

  return;
}
/******************************************************************************/

void geometry_plus (int ibc, int indx[], int *nu, int nsub, int node[]){
  /*
   * ===================================================================================================================
   *  Project 2 comments
   * ===================================================================================================================
   * Version: 1.0
   * So this function was the product of both frustration and time-constraints. There were two loops inside the original
   * geometry function which were very hard to work around my openmpi implementation. Version 1.0's implementation focuses on
   * breaking up the data arrays -> calling geometry with the sub arrays -> re-combining them.
   *
   * The two arrays being declare in this function couldn't have been done in the above manner because they depend on
   * previous input to produce current. So The two loops were broken off to allow for the others to benefit from the performance
   * boost provided by openmpi.
   */
  int i = 0;

  //printf ( "\n" );
  //printf ( "Subint  Left Node  Right Node\n" );
  //printf ( "\n" );
  for ( i = 0; i < nsub; i++ )
  {
    node[0+i*2] = i;
    node[1+i*2] = i + 1;
    //printf ( "  %8d  %8d  %8d\n", i+1, node[0+i*2], node[1+i*2] );
  }


  *nu = 0;

  i = 0;
  if ( ibc == 1 || ibc == 3 )
  {
    indx[i] = -1;
  }
  else
  {
    *nu = *nu + 1;
    indx[i] = *nu;
  }

  for ( i = 1; i < nsub; i++ )
  {
    *nu = *nu + 1;
    indx[i] = *nu;
  }

  i = nsub;
  if ( ibc == 2 || ibc == 3 )
  {
    indx[i] = -1;
  }
  else
  {
    *nu = *nu + 1;
    indx[i] = *nu;
  }

  //printf ( "\n" );

  //printf ( "\n" );
  //printf ( "  Node  Unknown\n" );
  //printf ( "\n" );
  for ( i = 0; i <= nsub; i++ )
  {

    //printf ( "  %8d  %8d\n", i, indx[i] );
  }
  return;
}






void init ( int *ibc, int *nquad, double *ul, double *ur, double *xl, 
  double *xr )

/******************************************************************************/


{

  *ibc = 1;
  *nquad = 1;
  *ul = 0.0;
  *ur = 1.0;
  *xl = 0.0;
  *xr = 1.0;

  //printf ( "\n" );
  //printf ( "  The equation is to be solved for\n" );
  //printf ( "  X greater than XL = %f\n", *xl );
  //printf ( "  and less than XR = %f\n", *xr );
  //printf ( "\n" );
  //printf ( "  The boundary conditions are:\n" );
  //printf ( "\n" );

  if ( *ibc == 1 || *ibc == 3 )
  {
    //printf ( "  At X = XL, U = %f\n", *ul );
  }
  else
  {
    //printf ( "  At X = XL, U' = %f\n", *ul );
  }

  if ( *ibc == 2 || *ibc == 3 )
  {
    //printf ( "  At X = XR, U = %f\n", *ur );
  }
  else
  {
    //printf ( "  At X = XR, U' = %f\n", *ur );
  }

  //printf ( "\n" );
  //printf ( "  Number of quadrature points per element is %d\n", *nquad );

  return;
}
/******************************************************************************/

void output ( double f[], int ibc, int indx[], int nsub, int nu, double ul, 
  double ur, double xn[] )

/******************************************************************************/
{
  int i;
  double u;

  //printf ( "\n" );
  //printf ( "  Computed solution coefficients:\n" );
  //printf ( "\n" );
  //printf ( "  Node    X(I)        U(X(I))\n" );
  //printf ( "\n" );

  for ( i = 0; i <= nsub; i++ )
  {

    if ( i == 0 )
    {
      if ( ibc == 1 || ibc == 3 )
      {
        u = ul;
      }
      else
      {
        u = f[indx[i]-1];
      }
    }

    else if ( i == nsub )
    {
      if ( ibc == 2 || ibc == 3 )
      {
        u = ur;
      }
      else
      {
        u = f[indx[i]-1];
      }
    }

    else
    {
      u = f[indx[i]-1];
    }

    //printf ( "  %8d  %8f  %14f\n", i, xn[i], u );
  }

  return;
}
/******************************************************************************/

void phi ( int il, double x, double *phii, double *phiix, double xleft, 
  double xrite )

/******************************************************************************/

{
  if ( xleft <= x && x <= xrite )
  {
    if ( il == 1 )
    {
      *phii = ( xrite - x ) / ( xrite - xleft );
      *phiix =         -1.0 / ( xrite - xleft );
    }
    else
    {
      *phii = ( x - xleft ) / ( xrite - xleft );
      *phiix = 1.0          / ( xrite - xleft );
    }
  }
/*
  If X is outside of the interval, just set everything to 0.
*/
  else
  {
    *phii  = 0.0;
    *phiix = 0.0;
  }

  return;
}
/******************************************************************************/

double pp ( double x )

/******************************************************************************/

{
  double value;

  value = 1.0;

  return value;
}
/******************************************************************************/

void prsys ( double adiag[], double aleft[], double arite[], double f[], 
  int nu )

/******************************************************************************/

{
  int i;

  //printf ( "\n" );
  //printf ( "Printout of tridiagonal linear system:\n" );
  //printf ( "\n" );
  //printf ( "Equation  ALEFT  ADIAG  ARITE  RHS\n" );
  //printf ( "\n" );

  for ( i = 0; i < nu; i++ )
  {
    //printf ( "  %8d  %14f  %14f  %14f  %14f\n",i + 1, aleft[i], adiag[i], arite[i], f[i] );
  }

  return;
}
/******************************************************************************/

double qq ( double x )

/******************************************************************************/

{
  double value;

  value = 0.0;

  return value;
}
/******************************************************************************/

void solve ( double adiag[], double aleft[], double arite[], double f[], 
  int nu )

/******************************************************************************/

{
  int i;

  arite[0] = arite[0] / adiag[0];

  for ( i = 1; i < nu - 1; i++ )
  {
    adiag[i] = adiag[i] - aleft[i] * arite[i-1];
    arite[i] = arite[i] / adiag[i];
  }
  adiag[nu-1] = adiag[nu-1] - aleft[nu-1] * arite[nu-2];

  f[0] = f[0] / adiag[0];
  for ( i = 1; i < nu; i++ )
  {
    f[i] = ( f[i] - aleft[i] * f[i-1] ) / adiag[i];
  }

  for ( i = nu - 2; 0 <= i; i-- )
  {
    f[i] = f[i] - arite[i] * f[i+1];
  }

  return;
}
/******************************************************************************/

void timestamp ( void )

/******************************************************************************/

{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  size_t len;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  len = strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  //printf ( "%s\n", time_buffer );

  return;
# undef TIME_SIZE
}

# include <stdlib.h>
# include <stdio.h>
# include <time.h>
# include <sys/time.h>
# include <mpi.h>
# include <omp.h>

int main ( int argc, char *argv[]  );
void assemble ( double adiag[], double aleft[], double arite[], double f[], 
  double h[], int indx[], int nl, int node[], int nu, int nquad, int nsub, 
  double ul, double ur, double xn[], double xquad[] , int offset , int chunksize);
double ff ( double x );


void geometry ( double h[], int ibc, int indx[], int nl, int node[], int nsub,
                int *nu, double xl, double xn[], double xquad[], double xr, int offset , int chunksize);


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

int main ( int argc, char *argv[]  )

/******************************************************************************/

{
  struct timeval start, end;
  gettimeofday(&start, NULL);

# define NSUB atoi(argv[1])
# define NL 10
# define  MASTER 0
/*
    * ==================================================
    * OPENMPI INITIALIZATION + related variable initialization
    */
  int numtasks;
  int taskid;
  int offset;
  int chunksize;

  int provided;
  MPI_Status status;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  //printf("%d\n",provided);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD,&taskid);

/*
   * ===================================================================================================================
   *  Project 2 comments
   * ===================================================================================================================
   * Version: 1.0
   * Below is the result of implementing openmpi to the project. A local and master copy of most of the variables
   * were necessary. I tried to be clever by only initializing the master variables inside the master process.
   * ===================================================================================================================
   *
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
  int *local_indx;
/*
  * ===================================================================================================================
  *  Project 2 comments
  * ===================================================================================================================
   * Initially below was a if statement to control the initialization of variables to save space. As these were early versions
   * of the project. There were issues and walls in learning the ins and outs of openmpi and its requirements.
   * As such to simplify things for the development of the project, global/local variables were created.
   *
   * Will be cleaned up in the final product.
  * ===================================================================================================================
 */
    //if(taskid == MASTER) {
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
    local_indx = (int *) malloc(sizeof(int) * (NSUB + 1));


  timestamp ( );

/*
   * ===================================================================================================================
   *  Project 2 comments
   * ===================================================================================================================
   * Version: 1.0
   * ___________________________
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
   *
   * Version: 1.1
   * ___________________________
   * Based on the poor results of version 1.0, we went back to the drawing board. We knew that overhead costs would play
   * a big role as it did in openmp (the first project). A desire to minimize the amount of unnecessary information transferring
   * across the network was huge. This meant that instead of having the code transferred back to the master process, the
   * processes would need to work in isolation for as long as possible. As such, the initialization overhead cost would
   * be distributed across two functions. Doing so would hopefully provide some performance improvements.
   *
   * What was done?
   *
   * We used the same work distribution system found in version 1.1 except we added assemble into the mix. Some alterations
   * to the declaration of assemble we required to allow for offset and chunksize data to be sent through. What this did was allow
   * for the processes to work independently, and combine the results.
   *
   * You might notice a lack of the geometry_plus function. This was because, after digging around assemble, we found it
   * unachievable to implement a pure MPI with only partial arrays of indx and node. The entire array of both were required
   * in any iteration cycle. Therefore, the initialization of these arrays were re-introduced to geometry and allowed
   * to iterate over their normal range. Since these arrays didnt vary across processes, there wasn't any need to combine/
   * compare them.
   *
   * Results:
   * VERY GOOD!!!!
   * Across all executions involving 2,4,6,8 processes, when NSUB a value of 100,000 performance gains were noticed.
   * The performance gains ranged from:
   * 100,000:     at least 34% improvement across all
   * 10,000,000:  at least 85% improvement across all
   *
   * Version: 1.2
   * ___________________________
   * This version of the code attempted to further combine methods to iterate over the same data size.
   * Since prsys iterated over size NU, it was appropriate to just allow it to iterate over its normal iteration range.
   * I wasn't expecting anything huge with this version.
   *
   * Results:
   * Showed performance improvements but less than that of version 1.1, showing that 1.2 was adding to the run-time of
   * the application. Due to its simplicity, there isn't much performance gains that could be made. In the final version,
   * Code from the first project will be used to improve the speed of the function by dividing by two.
   *
   * Simple performance improvement.
   *
   * There was a 1% improvement in performance over version 1.1 when nsub reaches 10,000,000
   *
   * Version: 1.3
   * ___________________________
   * This code version attempts to expand on the benefits of the last code version by implementing V3.0 from project 1 +
   * V1.2 from the current project.
   *
   * Result:
   * There was a similar performance improvement to that of version 1.2. There were moments where the current version
   * had better performance but there was only at most a 1% improvement in performance.
   *
   *
   * Version: 2.0
   * ___________________________
   * This version and its sub-versions will focus on a hybrid implementation of MPI. This will involve combining the best
   * OpenMP implementations from the first project with the best implementations of version 1.0. Due to the small performance
   * improvement, version 1.2 and 1.3 and have ignored for now and version 2.0 has been built off version 1.1 which
   * showed the greatest performance improvement. This version, besides focusing on the hybrid implementation of MPI + MP,
   * focuses on geometry again. Except this time looking to implement a multi-threaded solution.
   *
   * Results:
   * ===================================================================================================================
   */


  int i;

/*
  Initialize the data.
*/
/*
   * ===================================================================================================================
   *  Project 2 comments
   * ===================================================================================================================
   * Version: 1.0
   * ___________________________
   * The data arrays were broken up into large sections which will, in the future, be futher optimized by incorporating openmp.
   * Initially, to get the project started, I implemented a poor mechanism to part off any additional data which might have been missed
   * in the case where the NSUB/numtasks didn't produce a nice number. Improvement in distributing this additional data will be
   * achieved in later versions.
   *
   * Version: 1.1
   * ___________________________
   * The work distribution mechanism remains the same, it has proven to work effectively.
   * In each case found below, geometry and assemble are paired
   * ===================================================================================================================
  */


  init ( &ibc, &nquad, &ul, &ur, &xl, &xr );
  chunksize = (NSUB/numtasks);
  offset = (int)(chunksize*taskid);
    /*
* ===================================================================================================================
*  Project 2 comments
* ===================================================================================================================
* Version: 1.2
* ___________________________
* Another ser of chunksize/offset variables were created to handle the NU iterations
* ===================================================================================================================
*/
    //int chunksize_nu = (nu/numtasks);
    //int offset_nu = (int)(chunksize_nu*taskid);


  if(taskid != MASTER) {
    if(taskid+1 == numtasks){
      // This is the current code mechanism, if the current process is the last simply make the end put the value of NSUB.
      geometry(local_h,ibc,local_indx, NL,node, NSUB,&nu, xl, local_xn, local_xquad, xr, offset, NSUB);
      assemble(local_adiag, local_aleft, local_arite, local_f, local_h, local_indx, NL, node, nu, nquad,NSUB, ul, ur, local_xn, local_xquad,offset, NSUB);
    }else{
      geometry(local_h,ibc,local_indx, NL,node, NSUB,&nu, xl, local_xn, local_xquad, xr, offset, chunksize + offset);
      assemble(local_adiag, local_aleft, local_arite, local_f, local_h, local_indx, NL, node, nu, nquad,NSUB, ul, ur, local_xn, local_xquad,offset, chunksize + offset);

    }

  }else{
    geometry(local_h,ibc,local_indx, NL,node, NSUB,&nu, xl, local_xn, local_xquad, xr, offset, chunksize);
    assemble(local_adiag, local_aleft, local_arite, local_f, local_h, local_indx, NL, node, nu, nquad,NSUB, ul, ur, local_xn, local_xquad,offset, chunksize);

  }

/*
   * ===================================================================================================================
   *  Project 2 comments
   * ===================================================================================================================
   * Version: 1.0
   * ___________________________
   * Below consists of the three data loops that are contained in geometry that need to be recombined.
   *
   * Version: 1.1
   * ___________________________
   * Alterations to the code required past geometry changed when assemble was included into the mix. After assemble, it
   * can be noticed that the only arrays that needed to be transferred were:
   *
   *    - adiag
   *    - aleft
   *    - arite
   *    - f
   *    - indx
   *    - xn
   *
   * ===================================================================================================================
   */

  MPI_Reduce(local_adiag, adiag,NSUB+1,MPI_DOUBLE,MPI_SUM, MASTER, MPI_COMM_WORLD);
  MPI_Reduce(local_aleft, aleft,NSUB+1,MPI_DOUBLE,MPI_SUM, MASTER, MPI_COMM_WORLD);
  MPI_Reduce(local_arite, arite,NSUB+1,MPI_DOUBLE,MPI_SUM, MASTER, MPI_COMM_WORLD);

  MPI_Reduce(local_f, f,NSUB+1,MPI_DOUBLE,MPI_MAX, MASTER, MPI_COMM_WORLD);

  MPI_Reduce(local_indx, indx,NSUB+1,MPI_INT,MPI_MAX, MASTER, MPI_COMM_WORLD);
  MPI_Reduce(local_xn, xn,NSUB+1,MPI_DOUBLE,MPI_MAX, MASTER, MPI_COMM_WORLD);


  if(taskid == MASTER) {

    prsys(adiag, aleft, arite, f, nu);
/*

  Solve the linear system.
*/
      /*
   * ===================================================================================================================
   *  Project 2 comments
   * ===================================================================================================================
   * Similar to project 1, solve displays no potential to be optimized due to its nature of computation. Every element
   * /computation requires the previous element. therefore, this must happen in a linear order to maintain identical output
   * ===================================================================================================================
   */

    solve(adiag, aleft, arite, f, nu);
/*
  Print out the solution.
*/
      /*
* ===================================================================================================================
*  Project 2 comments
* ===================================================================================================================
* A decision was made to not attempt to parallelize output by using openmpi due to road block of solve. Data would have
* to be sent to the master, solved, then re-distributed across the network which would kill any sort of performance
* improvements already made.
* ===================================================================================================================
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

}
/******************************************************************************/

void assemble ( double adiag[], double aleft[], double arite[], double f[], 
  double h[], int indx[], int nl, int node[], int nu, int nquad, int nsub, 
  double ul, double ur, double xn[], double xquad[], int offset , int chunksize )

/******************************************************************************/
/*
   * ===================================================================================================================
   *  Project 2 comments
   * ===================================================================================================================
   * Version: 1.1
   * ___________________________
   * In a similar manner to what was done in version 2.0, the four for loops were combined together to produce a single for loop.
   * Details of version 2.0 can be found below:
   *
   *===================================================================================================================
*/
/*
 * ==============================================================================================
 * Project 1 comments
 * ==============================================================================================
 * Code version 2.0
 *
 * Below is the biggest potential for parallelization in this program.
 * The first thing I did to increase performance in this application was to combine four for loops into
 * a single loop that you see below containing the following lines of code:
 *
 *    f[i] = 0.0;
 *    adiag[i] = 0.0;
 *    aleft[i] = 0.0;
 *    arite[i] = 0.0;
 *
 * By its self it made some small improvements in the run time.
 * ===================================================================================================================
 */
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
/*
  Zero out the arrays that hold the coefficients of the matrix
  and the right hand side.
*/
  for ( i = 0; i < nsub; i++ )
  {
    f[i] = 0.0;
    adiag[i] = 0.0;
    aleft[i] = 0.0;
    arite[i] = 0.0;
  }

/*
  For interval number IE,
*/
    /*
   * ===================================================================================================================
   *  Project 2 comments
   * ===================================================================================================================
   * Version: 1.1
   * ___________________________
   * As already mentioned, assemble and geometry work together in this pure MPI implementation by both only working on
   * an assigned iteration range. This was achieved by simply changing the ie start and end value.
   *
   * ===================================================================================================================
*/
  for ( ie = offset; ie < chunksize; ie++ )
  {
    he = h[ie];
    xleft = xn[node[0+ie*2]];
    xrite = xn[node[1+ie*2]];
/*
  consider each quadrature point IQ,
*/
    for ( iq = 0; iq < nquad; iq++ )
    {
      xquade = xquad[ie];
/*
  and evaluate the integrals associated with the basis functions
  for the left, and for the right nodes.
*/
      for ( il = 1; il <= nl; il++ )
      {
        ig = node[il-1+ie*2];
        iu = indx[ig] - 1;

        if ( 0 <= iu )
        {
          phi ( il, xquade, &phii, &phiix, xleft, xrite );
          f[iu] = f[iu] + he * ff ( xquade ) * phii;
/*
  Take care of boundary nodes at which U' was specified.
*/
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
/*
  Evaluate the integrals that take a product of the basis
  function times itself, or times the other basis function
  that is nonzero in this interval.
*/
          for ( jl = 1; jl <= nl; jl++ )
          {
            jg = node[jl-1+ie*2];
            ju = indx[jg] - 1;

            phi ( jl, xquade, &phij, &phijx, xleft, xrite );

            aij = he * ( pp ( xquade ) * phiix * phijx
                       + qq ( xquade ) * phii  * phij   );
/*
  If there is no variable associated with the node, then it's
  a specified boundary value, so we multiply the coefficient
  times the specified boundary value and subtract it from the
  right hand side.
*/
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
/*
  Otherwise, we add the coefficient we've just computed to the
  diagonal, or left or right entries of row IU of the matrix.
*/
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

void geometry ( double h[], int ibc, int indx[], int nl, int node[], int nsub,
                int *nu, double xl, double xn[], double xquad[], double xr, int offset , int chunksize)

/******************************************************************************/
/******************************************************************************/
/*
 * ===================================================================================================================
 *  Project 2 comments
 * ===================================================================================================================
 * Version: 1.1
 * ___________________________
 * In a similar manner to assemble, the code of geometry was converted to only iterate over a pre-determined
 * range. This allowed for a reduction in the run-time due to a decrease in for loop iterations.
 *
 * Version: 2.0
 * ___________________________
 * As mentioned, this code version will be focusing on how to combine openmp and openmpi into a hybrid solution.
 * For this code version, geometry will be the star of the show. Version 1.0 of project 1 will be combined with the
 * code from version 1.1 of the current project.
 * ===================================================================================================================
*/


/*
* =====================================================================================================================
* Project 1 Comments
* =====================================================================================================================
*  Version 1.0:
*  ______________________________
*
*  Building off of version 0.2, Wrapped the whole function in a parallel region.
*  Had issues with earlier attempts where the output was being distorted and there were multiple print statements being
*  made.
*
*  This issue was resolved by using:
*      #pragma omp single
*      #pragma omp barrier (to make sure that the threads caught up to each other)
*
*  Results:
*  - Found that the data was looking a bit strange, this was due to the fact that I was parallelizing
*    for loops which make calculations based on previous iterations calculations. Because of this backwards calculation,
*    parallelization is not possible.
*  - without fprintf statements:
*    without the inconsistent fprintf statements, a performance gain was achieved when NSUB was greater than 100,000.
*     The performance improvement increased exponentially.
*/
{
int i;
/*
Set the value of XN, the locations of the nodes.
*/
  //printf ( "\n" );
  //printf ( "  Node      Location\n" );
  //printf ( "\n" );
#pragma omp parallel
  {
#pragma omp for
    for (i = offset; i <= chunksize; i++) {
      xn[i] = ((double) (nsub - i) * xl
               + (double) i * xr)
              / (double) (nsub);
      //printf ( "  %8d  %14f \n", i, xn[i] );
    }
/*
  Set the lengths of each subinterval.
*/

    //printf ( "\n" );
    //printf ( "Subint    Length\n" );
    //printf ( "\n" );
#pragma omp for
    for (i = offset; i < chunksize; i++) {
      h[i] = xn[i + 1] - xn[i];
      //printf ( "  %8d  %14f\n", i+1, h[i] );
    }
/*
  Set the quadrature points, each of which is the midpoint
  of its subinterval.
*/

    //printf ( "\n" );
    //printf ( "Subint    Quadrature point\n" );
    //printf ( "\n" );
#pragma omp for
    for (i = offset; i < chunksize; i++) {
      xquad[i] = 0.5 * (xn[i] + xn[i + 1]);
      //printf ( "  %8d  %14f\n", i+1, xquad[i] );
    }
    /*
    Set the value of NODE, which records, for each interval,
    the node numbers at the left and right.
  */
    //printf ( "\n" );
    //printf ( "Subint  Left Node  Right Node\n" );
    //printf ( "\n" );

        /*
* ===================================================================================================================
*  Project 2 comments
* ===================================================================================================================
* Version: 1.1
* ___________________________
* Below were the two arrays which were causing issues in version 1.0. The two arrays in question were:
*    - indx
*    - node
* Due to assemble requiring all variables stored inside these two arrays to be initialized, they had to keep their
* normal iteration range.
*
* ===================================================================================================================
*/
#pragma omp for
    for (i = 0; i < nsub; i++) {
      node[0 + i * 2] = i;
      node[1 + i * 2] = i + 1;
      //printf ( "  %8d  %8d  %8d\n", i+1, node[0+i*2], node[1+i*2] );
    }

    /*
    Starting with node 0, see if an unknown is associated with
    the node.  If so, give it an index.
  */
#pragma omp single
    {
      *nu = 0;
/*
  Handle first node.
*/

      if (ibc == 1 || ibc == 3) {
        indx[0] = -1;
      }
      else {
        *nu = *nu + 1;
        indx[0] = *nu;
      }


/*
  Handle nodes 1 through nsub-1
*/
    }
#pragma omp for
    for (i = 1; i < nsub; i++) {
      *nu = *nu + 1;
      indx[i] = *nu;
    }
#pragma omp single
    {

      if (ibc == 2 || ibc == 3) {
        indx[nsub] = -1;
      }
      else {
        *nu = *nu + 1;
        indx[nsub] = *nu;
      }
    }
/*
  Handle the last node.
/*/


    //printf ( "\n" );

    //printf ( "\n" );
    //printf ( "  Node  Unknown\n" );
    //printf ( "\n" );
#pragma omp for
    for (i = 0; i <= nsub; i++) {

      //printf ( "  %8d  %8d\n", i, indx[i] );
    }
  }
  return;
}
/******************************************************************************/







void init ( int *ibc, int *nquad, double *ul, double *ur, double *xl, 
  double *xr )


{
/*
  IBC declares what the boundary conditions are.
*/
  *ibc = 1;
/*
  NQUAD is the number of quadrature points per subinterval.
  The program as currently written cannot handle any value for
  NQUAD except 1.
*/
  *nquad = 1;
/*
  Set the values of U or U' at the endpoints.
*/
  *ul = 0.0;
  *ur = 1.0;
/*
  Define the location of the endpoints of the interval.
*/
  *xl = 0.0;
  *xr = 1.0;
/*
  Print out the values that have been set.
*/
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
/*
  If we're at the first node, check the boundary condition.
*/
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
/*
  If we're at the last node, check the boundary condition.
*/
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
/*
  Any other node, we're sure the value is stored in F.
*/
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
//printf ( "\n" );
//printf ( "Printout of tridiagonal linear system:\n" );
//printf ( "\n" );
//printf ( "Equation  ALEFT  ADIAG  ARITE  RHS\n" );
//printf ( "\n" );
/*
* ===================================================================================================================
*  Project 2 comments
* ===================================================================================================================
* Version: 1.2
* ___________________________
* Iteration range was altered to try and minimize the number of unnecessary iterations. Altered the declaration of
* prsys to allow for offset data and chunksize to be passed.
*
*
 Version: 1.3
* ___________________________
* This version combined code from project 1's v3.0 (description found below) and pure MPI implementation of v1.2
* ===================================================================================================================
*/
/*
 * ===============================================================================================================
 * Project 1 Comments
 * ===============================================================================================================
 * Version 3.0:
 * ____________________________
 * This function is a very basic function as it only contains a single for loop.
 * The room for optimisation is quite small as the work isn't very complex.
 * This first version I looked at how I could reduce the number of iterations and focus on
 * improving the linear performance.
 *
 * I broke the code up into two sections, even and odd values of nu. I hoped to half the iteration size of the for loop
 * to give some performance improvements.
 * ===================================================================================================================
 */
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
/*
  Carry out Gauss elimination on the matrix, saving information
  needed for the backsolve.
*/
  arite[0] = arite[0] / adiag[0];

  for ( i = 1; i < nu - 1; i++ )
  {
    adiag[i] = adiag[i] - aleft[i] * arite[i-1];
    arite[i] = arite[i] / adiag[i];
  }
  adiag[nu-1] = adiag[nu-1] - aleft[nu-1] * arite[nu-2];
/*
  Carry out the same elimination steps on F that were done to the
  matrix.
*/
  f[0] = f[0] / adiag[0];
  for ( i = 1; i < nu; i++ )
  {
    f[i] = ( f[i] - aleft[i] * f[i-1] ) / adiag[i];
  }
/*
  And now carry out the steps of "back substitution".
*/
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

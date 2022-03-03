#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/types.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <math.h>
int MAX_NUM= 100;
double **makeMatrix(int n);
int freeMatrix(int n, double **mat);
double **randomMatrix(int n, double **mat);
double **threadManagement(int n, int numThreads, double **m1, double **m2, double **mat);
double **matrixMult(int n, double **m1, double **m2, double **mat);
void *threadMatrixMult(void *args);
double dotProduct(int n, double *row, double *col);
double *getColumn(int n, double **mat, int col);
int printCol(int n, double *col);
int printRow(int n, double *row);
int printMatrix(int n, double **mat);
double sos(int n, double **m1, double **m2);
double **matCopy(int n, double **copy, double **paste);

/**
* The params struct contains the arguments to be passed to the threadMatrixMult callback function.
*/
typedef struct{
	int n;
	double **m1;
	double **m2;
	double **mat;
	int row;
	int col;
	int numDots;
}params;

/**
* The threadMatrixMult() function deals with matrix multiplication when using threads.
* The function takes in two random matrices and fills certain slots of
* a third matrix with the multiplied values of the other two matrices. The function that
* calls threadMatrixMult ensures that all of the slots of the matrix get filled. 
*/
void *threadMatrixMult(void *args){
	//grab the struct that contains the arguments and grab all the arguments
	params *param = (params *)args;
	int n = param->n;
	int row = param->row;
	int col = param->col;
	double **m1 = param->m1;
	double **m2 = param->m2;
	double **mat = param->mat;
	int numDots = param->numDots;

	//column is used to hold the column of the current slot that is being calculated
	double *column;
	//while there are still slots to fill
	while(numDots>0){
		column = getColumn(n,m2,col);
		mat[row][col] = dotProduct(n,m1[row],column);
		free(column);
		col++;
		//if you reach the end of a row, move on to the next one
		if(col==n){
			row++;
			col = 0;
		}
		numDots--;	
	}
	//terminate the calling thread
	pthread_exit(0);
}

/**
* the matrixmult program takes in a number of threads to spin and a size of a square matrix. It then generates
* two random matrices and multiplies them together both with and without using threads, reporting the
* speedup obtained by using threads.
*
* I worked with Danielle Dolan on this assignment. I used the following resource for information on the
* gettimeofday() function:
* https://www.techiedelight.com/find-execution-time-c-program/
*/ 
int main(int argc, char *argv[]){
	//check to make sure there is a correct number of inputs
	if(argc!=3){
		printf("I'm sorry! you chose an invalid number of inputs. The file must take in two inputs, \n");
		printf("the first being the number of threads and \n");
		printf("the second being the size of a square matrix. \n");
		return 1;
	}
	else{
		int numThreads = atoi(argv[1]);
		int n = atoi(argv[2]);
		
		//check to make sure the number of threads is valid
		if(numThreads<0){
			printf("Number of threads entered must be nonnegative. Running the program on 0 threads... \n");
			numThreads = 0;
		}
		//check to make sure matrix size is valid
		if(n<=0){
			printf("Matrix size must be positive. \n");
			return 1;
		}
	
		time_t t;
		srand((unsigned) time(&t));
		struct timeval start, end;
		//make and fill two random matrices of size n
		double **m1 = randomMatrix(n,makeMatrix(n));
		double **m2 = randomMatrix(n,makeMatrix(n));
		double **mat;
		int SCALE_FACTOR = 10;
	
    	printf("Multiplying random matrices of size %dx%d.\n",n,n);
    	//bestUnthread will hold the quickest matrixmult time
    	long bestUnthread;
    	//tempUnthread will hold the matrix calculated without using threads
    	double **tempUnthread = makeMatrix(n);

    	//create space for the matrix
    	mat = makeMatrix(n);
    	for(int i = 0; i<5; i++){
    		//start the clock
    		gettimeofday(&start, NULL);
    		mat = matrixMult(n,m1,m2,mat);
    		gettimeofday(&end, NULL); 
    		//end the clock

    		long seconds = (end.tv_sec - start.tv_sec);
    		long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);

    		//on the first runthrough, scale the bestUnthread to be in the range of the calculation times
    		//and copy the multiplied matrix result into the temp
    		if(i==0){
    			bestUnthread = SCALE_FACTOR*micros;
    			tempUnthread = matCopy(n,mat,tempUnthread);
    		}
    		//update the bestUnthread found so far
    		if(micros<bestUnthread){
    			bestUnthread = micros;
    		}    		
    	}
    	//free the space created for the matrix
    	freeMatrix(n, mat);

    	printf("Best time without threading: %ld microseconds\n",  bestUnthread);

    	if(numThreads >0){
    		//bestThread will hold the quickest matrix result time
    		long bestThread;
    		//tempThread will hold the matrix calculated using threads
    		double **tempThread = makeMatrix(n);

    		//create space for the matrix
    		mat = makeMatrix(n);
    		for(int i = 0; i<5; i++){
    			//start the clock
    			gettimeofday(&start, NULL);
    			mat = threadManagement(n,numThreads,m1,m2,mat);
    			gettimeofday(&end, NULL); 
    			//end the clock

    			long seconds = (end.tv_sec - start.tv_sec);
    			long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    			
    			//on the first runthrough, scale the bestThread to be in the range of the calculation times
    			//and copy the multiplied matrix result into the temp
    			if(i==0){
    				bestThread = SCALE_FACTOR*micros;
    				tempThread = matCopy(n,mat,tempThread);
    			}
    			//update the bestThread found so far
    			if(micros<bestThread){
    				bestThread = micros;    				
    			}
    		}
    		//free space created for the matrix
    		freeMatrix(n, mat);

    		printf("Best time with %d threads: %ld microseconds \n",numThreads,bestThread);
    		printf("Observed speedup is a factor of %f.\n",((double)bestUnthread/bestThread));
    		printf("Observed error is %.3f.\n",sos(n,tempUnthread,tempThread));
    		//free the temporary thread matrix
    		free(tempThread);
    	}
    	//free the randomized matrices and the temporary unthreaded matrix
    	free(m1);
		free(m2);
		free(tempUnthread);
	}
	return 0;
}

/**
* The sos() function calculates the sum of squares error between two matrices.
*
* @param n 		the size of the matrix
* @param m1 	matrix 1
* @param m2 	matrix 2
* @return the sum of squares 	
*/
double sos(int n, double **m1, double **m2){
	double sos = 0.0;
	//for each spot in the matrix, calculate their difference squared
	for(int i = 0; i<n; i++){
		for(int j = 0; j<n; j++){
			sos += pow((m1[i][j]-m2[i][j]),2.0);
		}
	}
	return sos;
}

/**
* The matCopy() function copies one matrix into another.
*
* @param n 		the size of the matrix
* @param copy 	the matrix to copy the values from
* @param paste  the matrix to copy the values into
* @return the copied matrix
*/
double **matCopy(int n, double **copy, double **paste){
	//copy each value in the matrix
	for(int i = 0; i<n; i++){
		for(int j = 0; j<n; j++){
			paste[i][j] = copy[i][j];
		}
	}
	return paste;
}

/**
* The threadManagement() function is responsible for multiplying two matrices together using threads.
* The function creates the given number of threads, assigns each thread a range
* of slots to fill in a matrix multiplication, and fills the matrix.
*
* @param n 			the size of the matrix
* @param numThreads	the number of threads to spin 	
* @param m1 		the first randomized matrix
* @param m2 		the second randomized matrix 
* @param mat 		the matrix that will hold the result of m1*m2
* @return mat
*
* I used the following resource for information on passing a callback function multiple arguments:
*https://stackoverflow.com/questions/1352749/multiple-arguments-to-function-called-by-pthread-create
*/
double **threadManagement(int n, int numThreads, double **m1, double **m2, double **mat){
	pthread_t *threads = (pthread_t*)malloc(sizeof(pthread_t) * numThreads);
	//make an array of structs to hold the arguments for the threadMatrixMult
	params *args[numThreads];

	//numDots is the number of dot products each thread must calculate
	int numDots = (int)(n*n)/numThreads;
	int remainder = n*n % numThreads;
	//row and col will hold the location of the first slot that each thread must fill
	int row = 0;
	int col = 0;
	//tempDots will be numDots + 1 if there is still a remainder
	int tempDots;

	// spin them off with args!
	for (int i=0; i<numThreads; i++) {
		//allocate space for the struct and fill its values
		args[i] = (params*)malloc(sizeof(params));
  		args[i]->n = n;
  		args[i]->m1 = m1;
  		args[i]->m2 = m2;
  		args[i]->mat = mat;
  		args[i]->row = row;
  		args[i]->col = col;
  		//disperse the remainder amongst the threads
  		if(remainder>0){
  			tempDots = numDots+1;
  			remainder--;
  		}
  		else{
  			tempDots = numDots;
  		}
  		args[i]->numDots = tempDots;
	
		//create a thread and call threadMatrixMult with the argument struct
  		pthread_create(&threads[i], NULL, threadMatrixMult, (void *)args[i]);

  		//increment the row and col to get them to the position of the next empty
  		//slot that will be filled by the next thread
  		row += (tempDots+col)/n;
  		col=(col+tempDots)%n;
  		//this check should never happen, but if it does, break
  		if(row>=n){
  			break;
  		}
  	}

  	// rejoin them, free data
	for (int i=0; i<numThreads; i++) {
  		pthread_join(threads[i], NULL);
  		free(args[i]);
	}
	free(threads);
	return mat;

}

/**
* The makeMatrix() function allocates space for a 2D array.
*
* @param n 		the size of the matrix
* @return a pointer to the free space
*/
double **makeMatrix(int n){
	double **mat = (double **)malloc(n*sizeof(double*));
	for(int i = 0; i<n; i++){
		mat[i] = (double*)malloc(n*sizeof(double));
	}	
	//printf("RAND_MAX: %d \n",RAND_MAX);
	return mat;
}

/**
* The freeMatrix()function frees the space given to a matrix.
*
* @param n 		the size of the matrix
* @param mat  	the matrix to free
* @return 0 if completed without error
*/
int freeMatrix(int n, double **mat){
	for(int i = 0; i<n; i++){
		free(mat[i]);
	}
	free(mat);
	return 0;
}

/**
* The randomMatrix() function fills in a matrix with random doubles from 0 to 99.
*
* @param n 		the size of the matrix
* @param mat 	the matrix to fill
* @return the filled matrix
*/
double **randomMatrix(int n, double **mat){
	//fill each slot with a random double
	for(int i = 0; i<n; i++){
		for(int j = 0; j<n; j++){
			mat[i][j] = (double)((rand()/(double)RAND_MAX)*MAX_NUM);
		}
	}
	return mat;
}

/**
* The matrixMult() fuction multiplies two matrices together by calculating the dot product
* of the row of the first matrix and the column of the second matrix for each slot.
*
* @param n 		the size of the matrix
* @param m1  	the first matrix
* @param m2 	the second matrix
* @param mat    the resulting matrix
* @return the result
*/
double **matrixMult(int n, double **m1, double **m2, double **mat){
	//col wil hold the current column
	double *col;
	for(int i = 0; i<n; i++){
		for( int j = 0; j<n; j++){
			//get the column, fill the matrix with the dot product, free the column
			col = getColumn(n,m2,j);
			mat[i][j] = dotProduct(n,m1[i],col);
			free(col);
		}
	}	
	return mat;
}

/**
* The dotProduct() function calculates the dot product between two vectors of the same size.
*
* @param n 		the size of the vectors
* @param row  	vector 1
* @param col 	vector 2
* @return the scalar product
*/
double dotProduct(int n, double *row, double *col){
	double prod = 0;
	//calculate the dot product
	for(int i = 0; i< n; i++){
		prod +=row[i]*col[i];
	}
	return prod;
}

/**
* The getColumn() function fills an array with the values down the column of a 2D array.
*
* @param n 		the size of the matrix
* @param mat    the matrix to grab the column from
* @param c 		the column to grab
* @return column c
*/
double *getColumn(int n, double **mat, int c){
	double *col = (double *)malloc(n*sizeof(double));
	for(int i = 0; i<n; i++){
		col[i] = mat[i][c];
	}
	return col;
}

/**
* The printMatrix() function prints a matrix.
*/
int printMatrix(int n, double **mat){
	for(int i = 0; i<n;i++){
		for(int j = 0; j<n; j++){
			printf("%f ",mat[i][j]);
		}
		printf("\n");

	}
	printf("\n");
	return 0;
}

/**
* The printRow() function prints a row.
*/
int printRow(int n, double *row){
for(int i = 0; i<n; i++){
		printf("%f \n",row[i]);
	}
	return 0;
}

/**
* The printCol() function prints a column.
*/
int printCol(int n, double *col){
	for(int i = 0; i<n; i++){
		printf("%f \n",col[i]);
	}
	return 0;
}
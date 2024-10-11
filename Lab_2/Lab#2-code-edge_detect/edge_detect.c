/* This program detects the edges in a 256 gray-level 128 x 128 pixel image.
   The program relies on a 2D-convolution routine to convolve the image with
   kernels (Sobel operators) that expose horizontal and vertical edge
   information.

   The following is a block diagram of the steps performed in edge detection,


            +---------+       +----------+
   Input    |Smoothing|       |Horizontal|-------+
   Image -->| Filter  |---+-->| Gradient |       |
            +---------+   |   +----------+  +----x-----+   +---------+  Binary
                          |                 | Gradient |   |  Apply  |  Edge
                          |                 | Combining|-->|Threshold|->Detected
                          |   +----------+  +----x-----+   +----x----+  Output
                          |   | Vertical |       |              |
                          +-->| Gradient |-------+              |
                              +----------+                   Threshold
                                                               Value


    This program is based on the routines and algorithms found in the book
    "C Language Algorithms for Digital Signal Processing" by P.M. Embree
    and B. Kimble.

    Copyright (c) 1992 -- Mazen A.R. Saghir -- University of Toronto */
/* Modified to use arrays - SMP */

//#include "traps.h"

/*
* This is a variant of the original edge detection benchmark with the following 
* modifications:
* - all computations were encapsulated in functions and the main exclusively 
* 	consists now of calls to functions
* - parameterized capabilities are now define parameters, being N the size
*	of the squared images used
*/


#define         K       3 // fixed value
#define         N       1024 //512 //128
#define         T       127 // fixed value

// for now we only use squared images
int image_buffer1[N][N];
int image_buffer2[N][N];
int image_buffer3[N][N];
int filter[K][K];

void convolve2d(int input_image[N][N], int kernel[K][K], int output_image[N][N]);

void initialize(int image_buffer2[N][N], int image_buffer3[N][N]);

void combthreshold(int image_buffer1[N][N], int image_buffer2[N][N], int image_buffer3[N][N]);

void initcoeff1(int filter[K][K]);

void initcoeff2(int filter[K][K]);

void initcoeff3(int filter[K][K]);
	
main()
{

  /* Read input image. */
  input_dsp(image_buffer1, N*N, 1);


  /* Initialize image_buffer2 and image_buffer3 */
  initialize(image_buffer2, image_buffer3);


/* Set the values of the filter matrix to a Gaussian kernel.
   This is used as a low-pass filter which blurs the image so as to
   de-emphasize the response of some isolated points to the edge
   detection (Sobel) kernels. */
 
  initcoeff1(filter);


  /* Perform the Gaussian convolution. */

  convolve2d(image_buffer1, filter, image_buffer3);

  /* Set the values of the filter matrix to the vertical Sobel operator. */

  initcoeff2(filter);
  
  /* Convolve the smoothed matrix with the vertical Sobel kernel. */

  convolve2d(image_buffer3, filter, image_buffer1);

  /* Set the values of the filter matrix to the horizontal Sobel operator. */

  initcoeff3(filter);
  
  /* Convolve the smoothed matrix with the horizontal Sobel kernel. */

  convolve2d(image_buffer3, filter, image_buffer2);

  /* Take the larger of the magnitudes of the horizontal and vertical
     matrices. Form a binary image by comparing to a threshold and
     storing one of two values. */
     
  combthreshold(image_buffer1, image_buffer2, image_buffer3);


  /* Store binary image. */
  //output_dsp(image_buffer1, N*N, 1);
  //output_dsp(image_buffer2, N*N, 1);
  output_dsp(image_buffer3, N*N, 1);
 // output_dsp(filter, K*K, 1);
}



void initialize(int image_buffer2[N][N], int image_buffer3[N][N]) {
     int i, j;

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
       image_buffer2[i][j] = 0;
       //printf("address: %d\n", &image_buffer2[i][j]);
       image_buffer3[i][j] = 0;
     }
  }
}

void initcoeff1(int filter[K][K]) {
  filter[0][0] = 1;
  filter[0][1] = 2;
  filter[0][2] = 1;
  filter[1][0] = 2;
  filter[1][1] = 4;
  filter[1][2] = 2;
  filter[2][0] = 1;
  filter[2][1] = 2;
  filter[2][2] = 1;
}

void initcoeff2(int filter[K][K]) {
  filter[0][0] =  1;
  filter[0][1] =  0;
  filter[0][2] = -1;
  filter[1][0] =  2;
  filter[1][1] =  0;
  filter[1][2] = -2;
  filter[2][0] =  1;
  filter[2][1] =  0;
  filter[2][2] = -1;
}

void initcoeff3(int filter[K][K]) {
  filter[0][0] =  1;
  filter[0][1] =  2;
  filter[0][2] =  1;
  filter[1][0] =  0;
  filter[1][1] =  0;
  filter[1][2] =  0;
  filter[2][0] = -1;
  filter[2][1] = -2;
  filter[2][2] = -1;
}

void combthreshold(int image_buffer1[N][N], int image_buffer2[N][N], int  image_buffer3[N][N]) {

  int i,j;

  int temp1;
  int temp2;
  int temp3;

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; ++j) {
       temp1 = abs(image_buffer1[i][j]);
       temp2 = abs(image_buffer2[i][j]);
       temp3 = (temp1 > temp2) ? temp1 : temp2;
       image_buffer3[i][j] = (temp3 > T) ? 255 : 0;
     }
  }
}

/* This function convolves the input image by the kernel and stores the result
   in the output image. */

void convolve2d(int input_image[N][N], int kernel[K][K], int output_image[N][N])
{
  int i;
  int j;
  int c;
  int r;
  int normal_factor;
  int sum;
  int dead_rows;
  int dead_cols;

  /* Set the number of dead rows and columns. These represent the band of rows
     and columns around the edge of the image whose pixels must be formed from
     less than a full kernel-sized compliment of input image pixels. No output
     values for these dead rows and columns since  they would tend to have less
     than full amplitude values and would exhibit a "washed-out" look known as
     convolution edge effects. */

  dead_rows = K / 2;
  dead_cols = K / 2;

  /* Calculate the normalization factor of the kernel matrix. */

  normal_factor = 0;
  for (r = 0; r < K; r++) {
    for (c = 0; c < K; c++) {
      normal_factor += abs(kernel[r][c]);
    }
  }

  if (normal_factor == 0)
    normal_factor = 1;

  /* Convolve the input image with the kernel. */
  for (r = 0; r < N - K + 1; r++) {
    for (c = 0; c < N - K + 1; c++) {
      sum = 0;
      for (i = 0; i < K; i++) {
        for (j = 0; j < K; j++) {
          sum += input_image[r+i][c+j] * kernel[i][j];
        }
      }
      output_image[r+dead_rows][c+dead_cols] = (sum / normal_factor);
    }
  }
}

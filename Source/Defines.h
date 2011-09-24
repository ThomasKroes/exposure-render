#pragma once

#include <host_defines.h>
#include <float.h>

#pragma warning(disable : 4244)
#pragma warning(disable : 4251)
#pragma warning(disable : 4297)

#define HO													__host__
#define DEV													__device__
#define HOD													__host__ __device__
#define CD													__device__ __constant__
#define PI_F												3.141592654f	
#define HALF_PI_F											0.5f * PI_F
#define QUARTER_PI_F										0.25f * PI_F
#define TWO_PI_F											2.0f * PI_F
#define INV_PI_F											0.31830988618379067154f
#define INV_TWO_PI_F										0.15915494309189533577f
#define INV_4_PI_F											1.0f / (4.0f * PI_F)
#define	EULER_F												2.718281828f
#define RAD_F												57.29577951308232f
#define TWO_RAD_F											2.0f * RAD_F
#define DEG_TO_RAD											1.0f / RAD_F
#define	INF_MIN												-FLT_MAX
#define	INF_MAX												FLT_MAX
#define	RAY_MIN												-100000.0f
#define	RAY_MAX												100000.0f
#define EULER_E_F											2.71828182845904523536f
#define MAX_CHAR_SIZE										128
#define MAX_CHAR_SIZE_FILE_TYPE								8
#define MAX_CHAR_SIZE_FILE_EXTENSION						8
#define MAX_CHAR_SIZE_FILE_NAME								8
#define MAX_CHAR_SIZE_FILE_PATH								128
#define NUM_ALLOCATION_SIZES								10
#define KERNEL												__global__
#define IMPORT_PROGRESS_UPDATE_INTERVAL						500
#define MAX_BXDFS											4
#define NO_HIT												-1
#define NO_NODE_ID											-1
#define HISTOGRAM_NUM_BINS									250
#define MDH_LINE_SIZE										500
#define WHITESPACE											" \t\n\r"
#define MAX_NO_TF_POINTS									20
#define MAX_NO_VOLUME_LIGHTS								3
#define	MAX_BOKEH_DATA										12
#define MB													powf(1024.0f, 2.0f)

#define PRINT_CUDA_ERROR(call) {                                    \
	cudaError err = call;                                                    \
	if( cudaSuccess != err) {                                                \
	Log("Cuda error in file '%s' in line %i : %s.\n",        \
	__FILE__, __LINE__, cudaGetErrorString( err) );              \
	exit(EXIT_FAILURE);                                                  \
	} }
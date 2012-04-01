
#include <host_defines.h>
#include <float.h>
#include <algorithm>
#include <math.h>

using namespace std;

namespace ExposureRender
{

#define KERNEL					__global__
#define HOST					__host__
#define DEVICE					__device__ __inline__
#define DEVICE_NI				DEVICE __noinline__
#define HOST_DEVICE				HOST DEVICE 
#define HOST_DEVICE_NI			HOST_DEVICE __noinline__
#define CD						__device__ __constant__
#define PI_F					3.141592654f	
#define HALF_PI_F				0.5f * PI_F
#define QUARTER_PI_F			0.25f * PI_F
#define TWO_PI_F				2.0f * PI_F
#define INV_PI_F				0.31830988618379067154f
#define INV_TWO_PI_F			0.15915494309189533577f
#define FOUR_PI_F				4.0f * PI_F
#define INV_FOUR_PI_F			1.0f / FOUR_PI_F
#define	EULER_F					2.718281828f
#define RAD_F					57.29577951308232f
#define TWO_RAD_F				2.0f * RAD_F
#define DEG_TO_RAD				1.0f / RAD_F
#define METRO_SIZE				256
#define	RAY_EPS					0.0001f
#define RAY_EPS_2				2.0f * RAY_EPS
#define ONE_OVER_6				1.0f / 2.0f
#define ONE_OVER_255			1.0f / 255.0f

}
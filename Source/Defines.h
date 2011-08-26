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
#define	EULER_F												2.718281828f
#define RAD_F												57.29577951308232f
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

#define PRINT_CUDA_ERROR(call) {                                    \
	cudaError err = call;                                                    \
	if( cudaSuccess != err) {                                                \
	Log("Cuda error in file '%s' in line %i : %s.\n",        \
	__FILE__, __LINE__, cudaGetErrorString( err) );              \
	exit(EXIT_FAILURE);                                                  \
	} }

// Scene
#define DEF_SCENE_STATUS									SceneStatusUndefined
#define DEF_SCENE_ENVIRONMENT_TYPE							EnvironmentTypeTexture
#define DEF_SCENE_ACCELERATOR_TYPE							AcceleratorTypeBVH

// Focus
#define DEF_FOCUS_TYPE										CenterScreen
#define DEF_FOCUS_SENSOR_POS_CANVAS							Vec2f(0.0f)
#define DEF_FOCUS_P											Vec3f(0.0f)
#define DEF_FOCUS_FOCAL_DISTANCE							100.0f
#define	DEF_FOCUS_T											0.0f
#define DEF_FOCUS_N											Vec3f(0.0f)
#define DEF_FOCUS_DOT_WN									0.0f

// Aperture
#define DEF_APERTURE_SIZE									0.0f
#define DEF_APERTURE_NO_BLADES								5
#define DEF_APERTURE_BIAS									BiasNone
#define DEF_APERTURE_ROTATION								0.0f

// Film
#define DEF_FILM_RESOLUTION									Vec2i(512, 512)

// Importance
#define DEF_IMPORTANCE_ENABLED								false
#define DEF_IMPORTANCE_RADIUS								100.0f
#define DEF_IMPORTANCE_PROBABILITY							0.5f

// Tone mapping
#define DEF_TONE_MAP_EXPOSURE								1.0f

// Camera
#define DEF_CAMERA_TYPE										Perspective
#define DEF_CAMERA_OPERATOR									CameraOperatorUndefined
#define DEF_CAMERA_VIEW_MODE								ViewModeBack
#define DEF_CAMERA_HITHER									1.0f
#define DEF_CAMERA_YON										50000.0f
#define DEF_CAMERA_ENABLE_CLIPPING							true
#define DEF_CAMERA_GAMMA									2.2f
#define DEF_CAMERA_FIELD_OF_VIEW							55.0f
#define DEF_CAMERA_NUM_APERTURE_BLADES						4
#define DEF_CAMERA_APERTURE_BLADES_ANGLE					0.0f
#define DEF_CAMERA_ASPECT_RATIO								1.0f
#define DEF_CAMERA_ZOOM_SPEED								1.0f
#define DEF_CAMERA_ORBIT_SPEED								5.0f
#define DEF_CAMERA_APERTURE_SPEED							0.25f
#define DEF_CAMERA_FOCAL_DISTANCE_SPEED						10.0f

// Default global render params
#define DEF_RENDER_PARAMS_GLOBAL_TYPE						RenderTypeMLT
#define DEF_RENDER_PARAMS_GLOBAL_RAY_EPSILON				0.01f
#define DEF_RENDER_PARAMS_GLOBAL_DEGRADE					false
#define DEF_RENDER_PARAMS_GLOBAL_DEGRADATION_LEVEL			1
#define DEF_RENDER_PARAMS_GLOBAL_RENDER_PASS				0
#define DEF_RENDER_PARAMS_GLOBAL_MAX_SPm_P					1024
#define DEF_RENDER_PARAMS_GLOBAL_PROGRESSIVE				true
#define DEF_RENDER_PARAMS_GLOBAL_NUM_SAMPLE_BINS			8
#define DEF_RENDER_PARAMS_SIGMA_A							0.5f
#define DEF_RENDER_PARAMS_SIGMA_S							0.5f
#define DEF_RENDER_PARAMS_LVE								SPEC_GRAY_50

// Daylight
#define DEF_ENVIRONMENT_DAYLIGHT_LONGITUDE					28.08f
#define DEF_ENVIRONMENT_DAYLIGHT_LATITUDE					-26.2f
#define DEF_ENVIRONMENT_DAYLIGHT_YEAR						2010
#define DEF_ENVIRONMENT_DAYLIGHT_MONTH						6
#define DEF_ENVIRONMENT_DAYLIGHT_DAY						25
#define DEF_ENVIRONMENT_DAYLIGHT_HOUR						12
#define DEF_ENVIRONMENT_DAYLIGHT_DAYLIGHT_SAVING_TIME		false
#define DEF_ENVIRONMENT_DAYLIGHT_MINUTE						1
#define DEF_ENVIRONMENT_DAYLIGHT_TIME_ZONE					0
#define DEF_ENVIRONMENT_DAYLIGHT_ALTITUDE					35.0f
#define DEF_ENVIRONMENT_DAYLIGHT_AZIMUTH					180.0f
#define DEF_ENVIRONMENT_DAYLIGHT_NORTH_OFFSET				0.0f
#define DEF_ENVIRONMENT_DAYLIGHT_MANUAL						true
#define DEF_ENVIRONMENT_DAYLIGHT_ORBITAL_SCALE				200.0f
#define DEF_ENVIRONMENT_DAYLIGHT_SUN_POSITION				Vec3f(1000000.0f, 1000000.0f, 1000000.0f)
#define DEF_ENVIRONMENT_DAYLIGHT_POWER						1.0f
#define DEF_ENVIRONMENT_DAYLIGHT_TURBIDITY					2.0f
#define DEF_ENVIRONMENT_DAYLIGHT_RELATIVE_SUN_SIZE			1.0f;
#define DEF_ENVIRONMENT_DAYLIGHT_SUN_RADIANCE				Spectrum(1.0f, 1.0f, 1.0f)

// Texture
#define DEF_TEXTURE_TYPE									Procedural
#define DEF_TEXTURE_NAME									"Untitled"
#define DEF_TEXTURE_WIDTH									-1
#define DEF_TEXTURE_HEIGHT									-1
#define DEF_TEXTURE_POWER									1.0f
#define DEF_TEXTURE_OFFSET									Vec3f(0.0f, 0.0f, 0.0f)
#define DEF_TEXTURE_TILING									Vec3f(1.0f, 1.0f, 1.0f)
#define DEF_TEXTURE_ANGLE									Vec3f(0.0f, 0.0f, 0.0f)
#define DEF_TEXTURE_GRID_OUTLINE_SIZE						0.1f
#define DEF_TEXTURE_DOTS_RADIUS								0.25f
#define DEF_TEXTURE_OFFSET_INCREMENT_SPEED_DRAG				0.001f
#define DEF_TEXTURE_TILING_INCREMENT_SPEED_DRAG				0.005f
#define DEF_TEXTURE_ANGLE_INCREMENT_SPEED_DRAG				0.0025f
#define DEF_TEXTURE_BITMAP_ID								-1
#define DEF_TEXTURE_MIRROR_X								false
#define DEF_TEXTURE_MIRROR_Y								false
#define DEF_TEXTURE_MIRROR_Z								false
#define DEF_TEXTURE_BUMP_STRENGTH							1.0f

// Bitmap
#define DEF_BITMAP_WIDTH									0
#define DEF_BITMAP_HEIGHT									0
#define DEF_BITMAP_TYPE										"Undefined"
#define DEF_BITMAP_NAME										"Undefined"
#define DEF_BITMAP_PATH										"Undefined"
#define DEF_BITMAP_SIZE										0
#define DEF_BITMAP_INTERPOLATION_TYPE						None

#define MAX_NODE_STACK_SIZE									64

// Material
#define DEF_MATERIAL_NAME									"Untitled"
#define DEF_MATERIAL_TYPE									MaterialTypeLambert
#define DEF_MATERIAL_EMISSION_GAIN							1.0f
#define DEF_MATERIAL_EMISSION_POWER							100.0f
#define DEF_MATERIAL_EMISSION_EFFICACY						0.17f
#define DEF_MATERIAL_IOR									1.5f
#define DEF_MATERIAL_ROUGHNESS								5.0f
#define DEF_MATERIAL_EXPONENT								100.0f
#define DEF_MATERIAL_ANISOTROPIC							false	
#define DEF_MATERIAL_EXPONENT_U								10.0f	
#define DEF_MATERIAL_EXPONENT_V								10.0f	

/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

enum EAcceleratorType
{
	AcceleratorTypeUndefined = 0,
	AcceleratorTypeKD,
	AcceleratorTypeBVH
};

enum EStatistics
{
	StatisticsFPS					= 0x0001,
	StatisticsNumVertices			= 0x0002,
	StatisticsNumVertexNormals		= 0x0004,
	StatisticsNumFaces				= 0x0008,
	StatisticsDisplayOptions		= 0x0010,
	StatisticsResolution			= 0x0020,
	StatisticsRaysPerSecond			= 0x0040,
	StatisticsMousePosition			= 0x0080,
	StatisticsCameraFieldOfView		= 0x0100,
	StatisticsCameraFocalDistance	= 0x0200,
	StatisticsCameraAperture		= 0x0400,
	StatisticsRenderType			= 0x0800,
	StatisticsDocumentPath			= 0x1000,
	StatisticsAll					= StatisticsFPS | StatisticsNumVertices | StatisticsNumVertexNormals | StatisticsNumFaces | StatisticsDisplayOptions | StatisticsResolution | StatisticsRaysPerSecond | StatisticsMousePosition | StatisticsCameraFieldOfView | StatisticsCameraFocalDistance | StatisticsCameraAperture | StatisticsRenderType | StatisticsDocumentPath
};

enum ESceneType
{
	SceneTypeUndefined = 0,
	SceneType3DS,
	SceneTypeOBJ
};

enum ESceneLoadError
{
	SceneLoaderErrorUndefined = 0,
	SceneLoaderErrorOK,
	SceneLoaderErrorNone,
	SceneLoaderErrorFileNameInvalid,
	SceneLoaderErrorLoadError
};

enum EMouseState
{
	MouseButtonUndefined		= 0x00000001L,
	MouseButtonDown				= 0x00000002L,
	MouseButtonDownLeft			= 0x00000004L,
	MouseButtonDownMiddle		= 0x00000008L,
	MouseButtonDownRight		= 0x00000010L,
	MouseWheelScroll			= 0x00000020L,
};

enum ECameraOperator
{
	CameraOperatorUndefined = 0,
	CameraOperatorOrbit,
	CameraOperatorPan,
	CameraOperatorZoom,
	CameraOperatorFieldOfView,
	CameraOperatorAperture,
	CameraOperatorFocalDistance,
	CameraOperatorProbeFocalDistance,
};

// View modes
enum EViewMode
{
	ViewModeUndefined = 0,
	
	ViewModeUser,

	// Normal views
	ViewModeFront,
	ViewModeBack,
	ViewModeLeft,
	ViewModeRight,
	ViewModeTop,
	ViewModeBottom,

	// Isometric views
	ViewModeIsometricFrontLeftTop,
	ViewModeIsometricFrontRightTop,
	ViewModeIsometricFrontLeftBottom,
	ViewModeIsometricFrontRightBottom,
	ViewModeIsometricBackLeftTop,
	ViewModeIsometricBackRightTop,
	ViewModeIsometricBackLeftBottom,
	ViewModeIsometricBackRightBottom
};

// Mesh object view columns
enum EMeshObjectsViewColumns
{
	MeshObjectsViewColumnName,
	MeshObjectsViewColumnMaterial,
	MeshObjectsViewColumnEmitter,
	MeshObjectsViewColumnFaceCount,
};

// Scene status
enum ESceneStatus
{
	SceneStatusUndefined	= 0x000,
 	SceneStatusSuspend		= 0x001,
	SceneStatusIdle			= 0x002,
	SceneStatusLoaded		= 0x004,
	SceneStatusIO			= 0x008,
 	SceneStatusLocked		= 0x010,
 	SceneStatusPaused		= 0x010,
// 	SceneStatus				= 0x020,
// 	SceneStatus				= 0x040,
// 	SceneStatus				= 0x080,
// 	SceneStatus				= 0x100,
// 	SceneStatus				= 0x200,
// 	SceneStatus				= 0x400,
};							

// Type of material
enum EMaterialType
{
	MaterialTypeUndefined = 0,
	MaterialTypeLambert,
	MaterialTypeDiffuse,
	MaterialTypeGlossy,
	MaterialTypeSpecular
};

// Type of environment
enum EEnvironmentType
{
	EnvironmentTypeDaylight,
	EnvironmentTypeTexture
};

// CUDA memory types
enum ECudaAllocationType
{
	CudaAllocationTypeUndefined				= 0x000,
	CudaAllocationTypeMiscellaneous			= 0x001,
	CudaAllocationTypeTexture				= 0x002,
	CudaAllocationTypeGeometry				= 0x004,
	CudaAllocationTypeAccelerator			= 0x008,
	CudaAllocationTypeKernel				= 0x010,
	CudaAllocationTypeFrameBuffer			= 0x020,
	CudaAllocationTypeAccumulationBuffer	= 0x040,
	CudaAllocationTypeRandom				= 0x080,
	CudaAllocationEmitter					= 0x100
};

// Memory size unit
enum EMemoryUnit
{
	MemoryUnitUndefined = 0,
	MemoryUnitBit,
	MemoryUnitByte,
	MemoryUnitKiloByte,
	MemoryUnitMegaByte,
	MemoryUnitGigaByte,
	MemoryUnitTeraByte
};

// Type of sampling
enum ESamplingType
{
	SamplingTypeUndefined = 0,
	SamplingTypeRandom,
	SamplingTypeStratified
};

// BSDF Declarations
enum BxDFType
{
	BSDF_REFLECTION   		= 1<<0,
	BSDF_TRANSMISSION 		= 1<<1,
	BSDF_DIFFUSE      		= 1<<2,
	BSDF_GLOSSY       		= 1<<3,
	BSDF_SPECULAR     		= 1<<4,
	BSDF_ALL_TYPES			= BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR,
	BSDF_ALL_REFLECTION		= BSDF_REFLECTION | BSDF_ALL_TYPES,
	BSDF_ALL_TRANSMISSION	= BSDF_TRANSMISSION | BSDF_ALL_TYPES,
	BSDF_ALL				= BSDF_ALL_REFLECTION | BSDF_ALL_TRANSMISSION
};

enum EDirty
{
	MaterialsDirty			= 0x00001,
	TexturesDirty			= 0x00002,
	CameraDirty				= 0x00004,
	GeometryDirty			= 0x00008,
	AccelerationDirty		= 0x00010,
	BitmapsDirty			= 0x00020,
	VolumeDirty				= 0x00040,
	FrameBufferDirty		= 0x00080,
	RenderParamsDirty		= 0x00100,
	VolumeDataDirty			= 0x00200,
	FilmResolutionDirty		= 0x00400,
	EnvironmentDirty		= 0x00800,
	FocusDirty				= 0x01000,
	LightsDirty				= 0x02000,
	BenchmarkDirty			= 0x04000,
	TransferFunctionDirty	= 0x08000,
	AnimationDirty			= 0x10000,
};

enum EContainment
{
	ContainmentNone,
	ContainmentPartial,
	ContainmentFull
};

enum EAxis
{
	AxisX = 0,
	AxisY,
	AxisZ,
	AxisUndefined
};
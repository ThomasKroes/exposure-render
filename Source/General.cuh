/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#ifdef _EXPORTING
	#define EXPOSURE_RENDER_DLL    __declspec(dllexport)
#else
	#define EXPOSURE_RENDER_DLL    __declspec(dllimport)
#endif

struct EXPOSURE_RENDER_DLL Interval
{
	float	m_Min;
	float	m_Max;
	float	m_Range;
	float	m_InvRange;

	void Set(float Range[2])
	{
		m_Min		= Range[0];
		m_Max		= Range[1];
		m_Range		= m_Max- m_Min;
		m_InvRange	= m_Range != 0.0f ? 1.0f / m_Range : 0.0f;
	}
};

struct EXPOSURE_RENDER_DLL _TransformMatrix
{
	float		NN[4][4];

	/*
	DEV Vec3f GetPosition()
	{
		return Vec3f(NN[0][3], NN[1][3], NN[2][3]);
	}

	DEV Vec3f GetScale()
	{
		return Vec3f(NN[3][0], NN[3][1], NN[3][2]);
	}

	DEV Vec3f GetU()
	{
		return Vec3f(NN[0][0], NN[1][0], NN[2][0]);
	}

	DEV Vec3f GetV()
	{
		return Vec3f(NN[0][1], NN[1][1], NN[2][1]);
	}

	DEV Vec3f GetW()
	{
		return Vec3f(NN[0][2], NN[1][2], NN[2][2]);
	}
	*/
};

struct EXPOSURE_RENDER_DLL _Volume
{
	int			m_Extent[3];
	float		m_InvExtent[3];
	float		m_MinAABB[3];
	float		m_MaxAABB[3];
	float		m_InvMinAABB[3];
	float		m_InvMaxAABB[3];
	float		m_Size[3];
	float		m_InvSize[3];
	float		m_IntensityMin;
	float		m_IntensityMax;
	float		m_IntensityRange;
	float		m_IntensityInvRange;
	float		m_StepSize;
	float		m_StepSizeShadow;
	float		m_DensityScale;
	float		m_GradientDelta;
	float		m_InvGradientDelta;
	float		m_GradientDeltaX[3];
	float		m_GradientDeltaY[3];
	float		m_GradientDeltaZ[3];
	float		m_Spacing[3];
	float		m_InvSpacing[3];
	float		m_GradientFactor;
	int			m_ShadingType;
	float		m_MacroCellSize[3];
};

struct EXPOSURE_RENDER_DLL _Camera
{
	int			m_FilmWidth;
	int			m_FilmHeight;
	int			m_FilmNoPixels;
	float		m_Pos[3];
	float		m_Target[3];
	float		m_Up[3];
	float		m_N[3];
	float		m_U[3];
	float		m_V[3];
	float		m_FocalDistance;
	float		m_ApertureSize;
	float		m_ClipNear;
	float		m_ClipFar;
	float		m_Screen[2][2];
	float		m_InvScreen[2];
	float		m_Exposure;
	float		m_InvExposure;
	float		m_Gamma;
	float		m_InvGamma;
	float		m_FOV;
};

struct EXPOSURE_RENDER_DLL _Light
{
	bool				m_Visible;
	int					m_Type;
	int					m_TextureType;
	bool				m_OneSided;
	int					m_ShapeType;
	float				m_Color[3];
	_TransformMatrix	m_TM;
	_TransformMatrix	m_InvTM;
	float				m_Size[3];
	float				m_Area;
	float				m_InnerRadius;
	float				m_OuterRadius;
};

struct EXPOSURE_RENDER_DLL _Lighting
{
	int					m_NoLights;
	_Light				m_Lights[32];
};

struct EXPOSURE_RENDER_DLL _ClippingObject
{
	int					m_ShapeType;
	float				m_Size[3];
	float				m_Radius;
	bool				m_Invert;
	float				m_MinIntensity;
	float				m_MaxIntensity;
	_TransformMatrix	m_TM;
	_TransformMatrix	m_InvTM;
};

struct EXPOSURE_RENDER_DLL _Clipping
{
	int					m_NoClippingObjects;
	_ClippingObject		m_ClippingObjects[32];
};

struct EXPOSURE_RENDER_DLL _ReflectionObject
{
	_TransformMatrix	m_TM;
	_TransformMatrix	m_InvTM;
};

struct EXPOSURE_RENDER_DLL _Reflections
{
	int					m_NoReflectionObjects;
	_ReflectionObject	m_ReflectionObjects[32];
};

struct EXPOSURE_RENDER_DLL _Denoise
{
	float	m_Enabled;
	float	m_WindowRadius;
	float	m_WindowArea;
	float	m_InvWindowArea;
	float	m_Noise;
	float	m_WeightThreshold;
	float	m_LerpThreshold;
	float	m_LerpC;
};

struct EXPOSURE_RENDER_DLL _Scattering
{
	float		m_NoIterations;
	float		m_InvNoIterations;
	bool		m_Shadows;
};

struct EXPOSURE_RENDER_DLL _Blur
{
	int			m_FilterWidth;
	float		m_FilterWeights[10];
};
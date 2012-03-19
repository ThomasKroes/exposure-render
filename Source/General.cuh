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

#include <cstdio>

namespace ExposureRender
{

#ifdef _EXPORTING
	#define EXPOSURE_RENDER_DLL    __declspec(dllexport)
#else
	#define EXPOSURE_RENDER_DLL    __declspec(dllimport)
#endif

#define	MAX_NO_TIMINGS	64
#define	MAX_CHAR_SIZE	256

struct EXPOSURE_RENDER_DLL Exception
{
	char	Title[MAX_CHAR_SIZE];
	char	Description[MAX_CHAR_SIZE];
	
	Exception(const char* pTitle, const char* pDescription)
	{
		sprintf_s(Title, MAX_CHAR_SIZE, "%s", pTitle);
		sprintf_s(Description, MAX_CHAR_SIZE, "%s", pDescription);
	}
};

struct EXPOSURE_RENDER_DLL KernelTiming
{
	char	Event[MAX_CHAR_SIZE];
	float	Duration;

	KernelTiming()
	{
		sprintf_s(this->Event, MAX_CHAR_SIZE, "Undefined");
		this->Duration = 0.0f;
	}

	KernelTiming(const char* pEvent, const float& Duration)
	{
		sprintf_s(this->Event, MAX_CHAR_SIZE, pEvent);
		this->Duration = Duration;
	}

	KernelTiming& operator = (const KernelTiming& Other)
	{
		sprintf_s(this->Event, MAX_CHAR_SIZE, Other.Event);
		this->Duration = Other.Duration;

		return *this;
	}
};

struct EXPOSURE_RENDER_DLL KernelTimings
{
	KernelTiming	Timings[MAX_NO_TIMINGS];
	int				NoTimings;

	KernelTimings& operator = (const KernelTimings& Other)
	{
		for (int i = 0; i < MAX_NO_TIMINGS; i++)
		{
			this->Timings[i] = Other.Timings[i];
		}

		this->NoTimings = Other.NoTimings;

		return *this;
	}

	void Add(const KernelTiming& KernelTiming)
	{
		this->Timings[this->NoTimings] = KernelTiming;
	}

	void Reset()
	{
		this->NoTimings = 0;
	}
};

struct EXPOSURE_RENDER_DLL Range
{
	float	Min;
	float	Max;
	float	Extent;
	float	Inv;
	
	void Set(float Range[2])
	{
		this->Min		= Range[0];
		this->Max		= Range[1];
		this->Extent	= this->Max - this->Min;
		this->Inv		= this->Extent != 0.0f ? 1.0f / this->Extent : 0.0f;
	}
};

struct EXPOSURE_RENDER_DLL Matrix44
{
	float				NN[4][4];
};

struct EXPOSURE_RENDER_DLL VolumeProperties
{
	int					Extent[3];
	float				InvExtent[3];
	float				MinAABB[3];
	float				MaxAABB[3];
	float				InvMinAABB[3];
	float				InvMaxAABB[3];
	float				Size[3];
	float				InvSize[3];
	float				Scale;
	Range				IntensityRange;
	Range				GradientMagnitudeRange;
	float				Spacing[3];
	float				InvSpacing[3];
	float				GradientDeltaX[3];
	float				GradientDeltaY[3];
	float				GradientDeltaZ[3];
};

struct EXPOSURE_RENDER_DLL Camera
{
	int					FilmWidth;
	int					FilmHeight;
	int					FilmNoPixels;
	float				Pos[3];
	float				Target[3];
	float				Up[3];
	float				N[3];
	float				U[3];
	float				V[3];
	float				FocalDistance;
	float				ApertureSize;
	float				ClipNear;
	float				ClipFar;
	float				Screen[2][2];
	float				InvScreen[2];
	float				Exposure;
	float				InvExposure;
	float				Gamma;
	float				InvGamma;
	float				FOV;
};

struct EXPOSURE_RENDER_DLL Shape
{
	Matrix44			TM;
	Matrix44			InvTM;
	bool				OneSided;
	int					Type;
	float				Color[3];
	float				Size[3];
	float				Area;
	float				InnerRadius;
	float				OuterRadius;
};

struct EXPOSURE_RENDER_DLL Light
{
	bool				Visible;
	Shape				Shape;
	float				Color[3];
};

struct EXPOSURE_RENDER_DLL Lights
{
	int					NoLights;
	Light				LightList[32];
};

struct EXPOSURE_RENDER_DLL Clipper
{
	Shape				Shape;
	bool				Invert;
};

struct EXPOSURE_RENDER_DLL Clippers
{
	int					NoClippers;
	Clipper				ClipperList[32];
};

struct EXPOSURE_RENDER_DLL Reflector
{
	Shape				Shape;
	float				DiffuseColor[3];
	float				SpecularColor[3];
	float				Glossiness;
	float				Ior;
};

struct EXPOSURE_RENDER_DLL Reflectors
{
	int					NoReflectors;
	Reflector			ReflectorList[32];
};

struct EXPOSURE_RENDER_DLL RenderSettings
{
	struct TraversalSettings
	{
		float				StepSize;
		float				StepSizeShadow;
		bool				Shadows;
		float				MaxShadowDistance;
	};

	struct ShadingSettings
	{
		int					Type;
		float				DensityScale;
		float				IndexOfReflection;
		bool				OpacityModulated;
		int					GradientComputation;
		float				GradientThreshold;
		float				GradientFactor;
	};

	TraversalSettings	Traversal;
	ShadingSettings		Shading;
};

struct EXPOSURE_RENDER_DLL Filtering
{
	struct GaussianFilterParameters
	{
		int		KernelRadius;
		float	Sigma;
	};

	struct BilateralFilterParameters
	{
		float	SigmaD;
		float	SigmaR;
	};

	GaussianFilterParameters	FrameEstimateFilter;
	BilateralFilterParameters	PostProcessingFilter;
};

}
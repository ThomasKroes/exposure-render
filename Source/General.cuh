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

template<class T> inline void SetArray(T V[], const T& Value, const int& Size) { for (int i = 0; i < Size; i++) V[i] = Value; }
template<class T> inline void SetArray3(T V[], const T& Value) { SetArray(V, Value, 3); }

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

	void Init()
	{
		sprintf_s(this->Event, MAX_CHAR_SIZE, "");
		this->Duration = 0.0f;
	}
};

struct EXPOSURE_RENDER_DLL KernelTimings
{
	int				NoTimings;
	KernelTiming	Timings[MAX_NO_TIMINGS];
	
	KernelTimings& operator = (const KernelTimings& Other)
	{
		for (int i = 0; i < MAX_NO_TIMINGS; i++)
			this->Timings[i] = Other.Timings[i];

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

	void Init()
	{
		this->NoTimings = 0;

		for (int i = 0; i < MAX_NO_TIMINGS; i++)
			this->Timings[i].Init();
	}
};

struct EXPOSURE_RENDER_DLL Range
{
	float	Min;
	float	Max;
	float	Extent;
	float	Inv;
	
#ifndef __CUDA_ARCH__
	Range()
	{
		this->Min 		= 0.0f;
		this->Max 		= 0.0f;
		this->Extent	= 0.0f;
		this->Inv		= 0.0f;
	}
#endif

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

#ifndef __CUDA_ARCH__
	Matrix44()
	{
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				this->NN[i][j] = i == j ? 1.0f : 0.0f;
	}
#endif
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
	float				Spacing[3];
	float				InvSpacing[3];
	float				GradientDeltaX[3];
	float				GradientDeltaY[3];
	float				GradientDeltaZ[3];
	Range				IntensityRange;
	Range				GradientMagnitudeRange;

#ifndef __CUDA_ARCH__ 
	VolumeProperties()
	{
		this->Extent[0]				= 1;
		this->Extent[1]				= 1;
		this->Extent[2]				= 1;
		this->InvExtent[0]			= 1.0f;
		this->InvExtent[1]			= 1.0f;
		this->InvExtent[2]			= 1.0f;
		this->MinAABB[0]			= -0.5f;
		this->MinAABB[1]			= -0.5f;
		this->MinAABB[2]			= -0.5f;
		this->MaxAABB[0]			= 0.5f;
		this->MaxAABB[1]			= 0.5f;
		this->MaxAABB[2]			= 0.5f;
		this->InvMinAABB[0]			= -2.0f;
		this->InvMinAABB[1]			= -2.0f;
		this->InvMinAABB[2]			= -2.0f;
		this->InvMaxAABB[0]			= 2.0f;
		this->InvMaxAABB[1]			= 2.0f;
		this->InvMaxAABB[2]			= 2.0f;
		this->Size[0]				= 1.0f;
		this->Size[1]				= 1.0f;
		this->Size[2]				= 1.0f;
		this->InvSize[0]			= 1.0f;
		this->InvSize[1]			= 1.0f;
		this->InvSize[2]			= 1.0f;
		this->Spacing[0]			= 0.01f;
		this->Spacing[1]			= 0.01f;
		this->Spacing[2]			= 0.01f;
		this->InvSpacing[0]			= 100.0f;
		this->InvSpacing[1]			= 100.0f;
		this->InvSpacing[2]			= 100.0f;
		this->GradientDeltaX[0]		= 1.0f;
		this->GradientDeltaX[1]		= 0.0f;
		this->GradientDeltaX[2]		= 0.0f;
		this->GradientDeltaY[0]		= 0.0f;
		this->GradientDeltaY[1]		= 1.0f;
		this->GradientDeltaY[2]		= 0.0f;
		this->GradientDeltaZ[0]		= 0.0f;
		this->GradientDeltaZ[1]		= 0.0f;
		this->GradientDeltaZ[2]		= 1.0f;
	}
#endif
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

#ifndef __CUDA_ARCH__
	Camera()
	{
		this->FilmWidth			= 0;
		this->FilmHeight		= 0;
		this->FilmNoPixels		= 0;
		this->Pos[0]			= 0.0f;
		this->Pos[1]			= 0.0f;
		this->Pos[2]			= 0.0f;
		this->Target[0]			= 0.0f;
		this->Target[1]			= 0.0f;
		this->Target[2]			= 0.0f;
		this->Up[0]				= 0.0f;
		this->Up[1]				= 0.0f;
		this->Up[2]				= 0.0f;
		this->N[0]				= 0.0f;
		this->N[1]				= 0.0f;
		this->N[2]				= 0.0f;
		this->U[0]				= 0.0f;
		this->U[1]				= 0.0f;
		this->U[2]				= 0.0f;
		this->V[0]				= 0.0f;
		this->V[1]				= 0.0f;
		this->V[2]				= 0.0f;
		this->FocalDistance		= 0.0f;
		this->ApertureSize		= 0.0f;
		this->ClipNear			= 0.0f;
		this->ClipFar			= 0.0f;
		this->Screen[0][0]		= 0.0f;
		this->Screen[0][1]		= 0.0f;
		this->Screen[1][0]		= 0.0f;
		this->Screen[1][1]		= 0.0f;
		this->InvScreen[0]		= 0.0f;
		this->InvScreen[1]		= 0.0f;
		this->Exposure			= 0.0f;
		this->InvExposure		= 0.0f;
		this->Gamma				= 0.0f;
		this->InvGamma			= 0.0f;
		this->FOV				= 0.0f;
	}
#endif
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

#ifndef __CUDA_ARCH__
	Shape()
	{
		this->OneSided		= false;
		this->Type			= 0;
		this->Color[0]		= 0.0f;
		this->Color[1]		= 0.0f;
		this->Color[2]		= 0.0f;
		this->Size[0]		= 0.0f;
		this->Size[1]		= 0.0f;
		this->Size[2]		= 0.0f;
		this->Area			= 0.0f;
		this->InnerRadius	= 0.0f;
		this->OuterRadius	= 0.0f;
	}
#endif
};

struct EXPOSURE_RENDER_DLL Light
{
	bool	Visible;
	Shape	Shape;
	float	Color[3];

#ifndef __CUDA_ARCH__
	Light()
	{
		this->Visible	= true;
		this->Color[0]	= 1.0f;
		this->Color[1]	= 1.0f;
		this->Color[2]	= 1.0f;
	}
#endif
};

#define MAX_NO_LIGHTS 32

struct EXPOSURE_RENDER_DLL Lights
{
	int					NoLights;
	Light				LightList[MAX_NO_LIGHTS];

#ifndef __CUDA_ARCH__
	Lights()
	{
		this->NoLights	= 0;
	}
#endif
};

#define MAX_NO_TEXTURES 64

struct EXPOSURE_RENDER_DLL Texture
{
	struct EXPOSURE_RENDER_DLL Image
	{
		unsigned char*	pData;
		int				Size[2];

		Image& operator = (const Image& Other)
		{
			this->pData		= Other.pData;
			this->Size[0]	= Other.Size[0];
			this->Size[1]	= Other.Size[1];

			return *this;
		}

#ifndef __CUDA_ARCH__
		Image()
		{
			this->pData		= NULL;
			this->Size[0]	= 0;
			this->Size[1]	= 0;
		}
#endif
	};

	int			ID;
	int			Type;
	Image		Image;

#ifndef __CUDA_ARCH__
	Texture()
	{
		this->ID	= 0;
		this->Type	= 0;
	}
#endif

	Texture& operator = (const Texture& Other)
	{
		this->ID	= Other.ID;
		this->Type	= Other.Type;
		this->Image	= Other.Image;

		return *this;
	}
};

struct EXPOSURE_RENDER_DLL Textures
{
	int					NoTextures;
	Texture				TextureList[MAX_NO_TEXTURES];

#ifndef __CUDA_ARCH__
	Textures()
	{
		this->NoTextures = 0;
	}
#endif
};

struct EXPOSURE_RENDER_DLL Clipper
{
	Shape				Shape;
	bool				Invert;

#ifndef __CUDA_ARCH__
	Clipper()
	{
		this->Invert = false;
	}
#endif
};

#define MAX_NO_CLIPPERS 32

struct EXPOSURE_RENDER_DLL Clippers
{
	int					NoClippers;
	Clipper				ClipperList[MAX_NO_CLIPPERS];

#ifndef __CUDA_ARCH__
	Clippers()
	{
		this->NoClippers = 0;
	}
#endif
};

struct EXPOSURE_RENDER_DLL Reflector
{
	Shape				Shape;
	float				DiffuseColor[3];
	float				SpecularColor[3];
	float				Glossiness;
	float				Ior;

#ifndef __CUDA_ARCH__
	Reflector()
	{
		this->DiffuseColor[0]	= 0.0f;
		this->DiffuseColor[1]	= 0.0f;
		this->DiffuseColor[2]	= 0.0f;
		this->SpecularColor[0]	= 0.0f;
		this->SpecularColor[1]	= 0.0f;
		this->SpecularColor[2]	= 0.0f;
		this->Glossiness		= 0.0f;
		this->Ior				= 0.0f;
	}
#endif
};

#define MAX_NO_REFLECTORS 32

struct EXPOSURE_RENDER_DLL Reflectors
{
	int					NoReflectors;
	Reflector			ReflectorList[MAX_NO_REFLECTORS];

#ifndef __CUDA_ARCH__
	Reflectors()
	{
		this->NoReflectors = 0;
	}
#endif
};

struct EXPOSURE_RENDER_DLL RenderSettings
{
	struct EXPOSURE_RENDER_DLL TraversalSettings
	{
		float				StepSize;
		float				StepSizeShadow;
		bool				Shadows;
		float				MaxShadowDistance;

#ifndef __CUDA_ARCH__
		TraversalSettings()
		{
			this->StepSize			= 0.1f;
			this->StepSizeShadow	= 0.1f;
			this->Shadows			= true;
			this->MaxShadowDistance	= 1.0f;
		}
#endif
	};

	struct EXPOSURE_RENDER_DLL ShadingSettings
	{
		int					Type;
		float				DensityScale;
		float				IndexOfReflection;
		bool				OpacityModulated;
		int					GradientComputation;
		float				GradientThreshold;
		float				GradientFactor;

#ifndef __CUDA_ARCH__
		ShadingSettings()
		{
			this->Type					= 0;
			this->DensityScale			= 100.0f;
			this->IndexOfReflection		= 5.0f;
			this->OpacityModulated		= true;
			this->GradientComputation	= 1;
			this->GradientThreshold		= 0.5f;
			this->GradientFactor		= 0.5f;
		}
#endif
	};

	TraversalSettings	Traversal;
	ShadingSettings		Shading;

#ifndef __CUDA_ARCH__
	RenderSettings()
	{
	}
#endif
};

struct EXPOSURE_RENDER_DLL Filtering
{
	struct EXPOSURE_RENDER_DLL GaussianFilterParameters
	{
		int		KernelRadius;
		float	Sigma;

#ifndef __CUDA_ARCH__
		GaussianFilterParameters()
		{
			this->KernelRadius	= 2;
			this->Sigma			= 1.25f;
		}
#endif
	};

	struct EXPOSURE_RENDER_DLL BilateralFilterParameters
	{
		float	SigmaD;
		float	SigmaR;

#ifndef __CUDA_ARCH__
		BilateralFilterParameters()
		{
			this->SigmaD	= 5.0f;
			this->SigmaR	= 5.0f;
		}
#endif
	};

	GaussianFilterParameters	FrameEstimateFilter;
	BilateralFilterParameters	PostProcessingFilter;

#ifndef __CUDA_ARCH__
	Filtering()
	{
	}
#endif
};

}
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

struct EXPOSURE_RENDER_DLL ErRange
{
	float	Min;
	float	Max;
	float	Range;
	float	Inv;
	
	void Set(float Range[2])
	{
		this->Min		= Range[0];
		this->Max		= Range[1];
		this->Range		= this->Max - this->Min;
		this->Inv		= this->Range != 0.0f ? 1.0f / this->Range : 0.0f;
	}
};

struct EXPOSURE_RENDER_DLL ErMatrix44
{
	float				NN[4][4];
};

struct EXPOSURE_RENDER_DLL ErVolume
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
	ErRange				IntensityRange;
	ErRange				GradientMagnitudeRange;
	float				GradientThreshold;
	float				StepSize;
	float				StepSizeShadow;
	float				DensityScale;
	float				GradientDeltaX[3];
	float				GradientDeltaY[3];
	float				GradientDeltaZ[3];
	float				Spacing[3];
	float				InvSpacing[3];
	float				GradientFactor;
	int					ShadingType;
};

struct EXPOSURE_RENDER_DLL ErCamera
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

struct EXPOSURE_RENDER_DLL ErShape
{
	ErMatrix44			TM;
	ErMatrix44			InvTM;
	bool				OneSided;
	int					Type;
	float				Color[3];
	float				Size[3];
	float				Area;
	float				InnerRadius;
	float				OuterRadius;
};

struct EXPOSURE_RENDER_DLL ErLight
{
	bool				Visible;
	ErShape				Shape;
	float				Color[3];
};

struct EXPOSURE_RENDER_DLL ErLights
{
	int					NoLights;
	ErLight				LightList[32];
};

struct EXPOSURE_RENDER_DLL ErClipper
{
	ErShape				Shape;
	bool				Invert;
};

struct EXPOSURE_RENDER_DLL ErClippers
{
	int					NoClippers;
	ErClipper			ClipperList[32];
};

struct EXPOSURE_RENDER_DLL ErReflector
{
	ErShape				Shape;
	float				DiffuseColor[3];
	float				SpecularColor[3];
	float				Glossiness;
	float				Ior;
};

struct EXPOSURE_RENDER_DLL ErReflectors
{
	int					NoReflectors;
	ErReflector			ReflectorList[32];
};

struct EXPOSURE_RENDER_DLL ErDenoise
{
	float				Enabled;
	float				WindowRadius;
	float				WindowArea;
	float				InvWindowArea;
	float				Noise;
	float				WeightThreshold;
	float				LerpThreshold;
	float				LerpC;
};

struct EXPOSURE_RENDER_DLL ErScattering
{
	int					NoIterations;
	float				InvNoIterations;
	int					SamplingStrategy;
	float				IndexOfReflection;
	bool				OpacityModulated;
	bool				Shadows;
	float				MaxShadowDistance;
	int					GradientComputation;
};

struct EXPOSURE_RENDER_DLL ErBlur
{
	int					FilterWidth;
	float				FilterWeights[10];
};
/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or witDEVut modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software witDEVut specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT DEVLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT DEVLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) DEVWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "Defines.h"
#include "Enums.h"

namespace ExposureRender
{

struct EXPOSURE_RENDER_DLL RenderSettings
{
	struct EXPOSURE_RENDER_DLL TraversalSettings
	{
		float				StepSize;
		float				StepSizeShadow;
		bool				Shadows;
		float				MaxShadowDistance;

		TraversalSettings()
		{
			this->StepSize			= 0.1f;
			this->StepSizeShadow	= 0.1f;
			this->Shadows			= true;
			this->MaxShadowDistance	= 1.0f;
		}

		~TraversalSettings()
		{
		}

		TraversalSettings& operator = (const TraversalSettings& Other)
		{
			this->StepSize				= Other.StepSize;
			this->StepSizeShadow		= Other.StepSizeShadow;
			this->Shadows				= Other.Shadows;
			this->MaxShadowDistance		= Other.MaxShadowDistance;

			return *this;
		}
	};

	struct EXPOSURE_RENDER_DLL ShadingSettings
	{
		int					Type;
		float				DensityScale;
		bool				OpacityModulated;
		int					GradientComputation;
		float				GradientThreshold;
		float				GradientFactor;

		ShadingSettings()
		{
			this->Type					= 0;
			this->DensityScale			= 100.0f;
			this->OpacityModulated		= true;
			this->GradientComputation	= 1;
			this->GradientThreshold		= 0.5f;
			this->GradientFactor		= 0.5f;
		}

		~ShadingSettings()
		{
		}

		ShadingSettings& operator = (const ShadingSettings& Other)
		{
			this->Type					= Other.Type;
			this->DensityScale			= Other.DensityScale;
			this->OpacityModulated		= Other.OpacityModulated;
			this->GradientComputation	= Other.GradientComputation;
			this->GradientThreshold		= Other.GradientThreshold;
			this->GradientFactor		= Other.GradientFactor;

			return *this;
		}
	};

	TraversalSettings	Traversal;
	ShadingSettings		Shading;

	RenderSettings()
	{
	}

	~RenderSettings()
	{
	}

	RenderSettings& operator = (const RenderSettings& Other)
	{
		this->Traversal	= Other.Traversal;
		this->Shading	= Other.Shading;

		return *this;
	}
};

}
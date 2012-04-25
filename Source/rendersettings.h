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

#include "defines.h"
#include "enums.h"

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

		HOST TraversalSettings()
		{
			this->StepSize			= 0.1f;
			this->StepSizeShadow	= 0.1f;
			this->Shadows			= true;
			this->MaxShadowDistance	= 1.0f;
		}

		HOST ~TraversalSettings()
		{
		}
		
		HOST TraversalSettings(const TraversalSettings& Other)
		{
			*this = Other;
		}

		HOST TraversalSettings& operator = (const TraversalSettings& Other)
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

		HOST ShadingSettings()
		{
			this->Type					= 0;
			this->DensityScale			= 100.0f;
			this->OpacityModulated		= true;
			this->GradientComputation	= 1;
			this->GradientThreshold		= 0.5f;
			this->GradientFactor		= 0.5f;
		}

		HOST ~ShadingSettings()
		{
		}
		
		HOST ShadingSettings(const ShadingSettings& Other)
		{
			*this = Other;
		}

		HOST ShadingSettings& operator = (const ShadingSettings& Other)
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

	struct EXPOSURE_RENDER_DLL Filtering
	{
		struct EXPOSURE_RENDER_DLL GaussianFilterParameters
		{
			int		KernelRadius;
			float	Sigma;

			HOST GaussianFilterParameters()
			{
				this->KernelRadius	= 2;
				this->Sigma			= 1.25f;
			}

			HOST ~GaussianFilterParameters()
			{
			}

			HOST GaussianFilterParameters(const GaussianFilterParameters& Other)
			{
				*this = Other;
			}

			HOST GaussianFilterParameters& operator = (const GaussianFilterParameters& Other)
			{
				this->KernelRadius	= Other.KernelRadius;
				this->Sigma			= Other.Sigma;

				return *this;
			}
		};

		struct EXPOSURE_RENDER_DLL BilateralFilterParameters
		{
			float	SigmaD;
			float	SigmaR;

			HOST BilateralFilterParameters()
			{
				this->SigmaD	= 5.0f;
				this->SigmaR	= 5.0f;
			}
			
			HOST ~BilateralFilterParameters()
			{
			}

			HOST BilateralFilterParameters(const BilateralFilterParameters& Other)
			{
				*this = Other;
			}

			HOST BilateralFilterParameters& operator = (const BilateralFilterParameters& Other)
			{
				this->SigmaD	= Other.SigmaD;
				this->SigmaR	= Other.SigmaR;

				return *this;
			}
		};

		// FIXME
		GaussianFilterParameters	FrameEstimateFilterParams;
		BilateralFilterParameters	PostProcessingFilterParams;

		HOST Filtering()
		{
		}

		HOST ~Filtering()
		{
		}
		
		HOST Filtering(const Filtering& Other)
		{
			*this = Other;
		}

		HOST Filtering& operator = (const Filtering& Other)
		{
			this->FrameEstimateFilterParams		= Other.FrameEstimateFilterParams;
			this->PostProcessingFilterParams	= Other.PostProcessingFilterParams;

			return *this;
		}
	};

	TraversalSettings	Traversal;
	ShadingSettings		Shading;
	Filtering			Filtering;

	HOST RenderSettings()
	{
	}

	HOST ~RenderSettings()
	{
	}

	HOST RenderSettings(const RenderSettings& Other)
	{
		*this = Other;
	}

	HOST RenderSettings& operator = (const RenderSettings& Other)
	{
		this->Traversal		= Other.Traversal;
		this->Shading		= Other.Shading;
		this->Filtering		= Other.Filtering;

		return *this;
	}
};

}

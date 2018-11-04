/*
    Exposure Render: An interactive photo-realistic volume rendering framework
    Copyright (C) 2011 Thomas Kroes

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

#include "defines.h"
#include "enums.h"

namespace ExposureRender
{

class EXPOSURE_RENDER_DLL RenderSettings
{
public:
	class EXPOSURE_RENDER_DLL TraversalSettings
	{
	public:
		HOST TraversalSettings()
		{
			this->StepFactorPrimary	= 0.1f;
			this->StepFactorShadow	= 0.1f;
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
			this->StepFactorPrimary		= Other.StepFactorPrimary;
			this->StepFactorShadow		= Other.StepFactorShadow;
			this->Shadows				= Other.Shadows;
			this->MaxShadowDistance		= Other.MaxShadowDistance;

			return *this;
		}

		float	StepFactorPrimary;
		float	StepFactorShadow;
		bool	Shadows;
		float	MaxShadowDistance;
	};

	class EXPOSURE_RENDER_DLL ShadingSettings
	{
	public:
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

		int		Type;
		float	DensityScale;
		bool	OpacityModulated;
		int		GradientComputation;
		float	GradientThreshold;
		float	GradientFactor;
	};

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

		return *this;
	}

	TraversalSettings	Traversal;
	ShadingSettings		Shading;
};

}

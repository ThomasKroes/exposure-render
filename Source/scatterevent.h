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

#include "color.h"
#include "ray.h"
#include "matrix.h"
#include "shader.h"

using namespace std;

namespace ExposureRender
{

class ScatterEvent
{
public:
	HOST_DEVICE ScatterEvent()
	{
	}

	HOST_DEVICE ScatterEvent(const Enums::ScatterType& Type)
	{
		this->Type = Type;
		this->SetInvalid();
	}

	HOST_DEVICE void SetValid(float T, Vec3f P, Vec3f N, Vec3f Wo, ColorXYZf Le, Vec2f UV = Vec2f(0.0f))
	{
		this->Valid		= true;
		this->T			= T;
		this->P			= P;
		this->N			= N;
		this->Wo		= Wo;
		this->Le		= Le;
		this->UV		= UV;
	}

	HOST_DEVICE void SetInvalid()
	{
		this->Valid		= false;
		this->T			= 0.0f;
		this->P			= Vec3f();
		this->N			= Vec3f();
		this->Wo		= Vec3f();
		this->Le		= ColorXYZf(0.0f);
		this->UV		= Vec2f(0.0f);
	}

	HOST_DEVICE ScatterEvent& ScatterEvent::operator = (const ScatterEvent& Other)
	{
		this->Type			= Other.Type;
		this->Valid			= Other.Valid;
		this->T				= Other.T;
		this->P				= Other.P;
		this->N				= Other.N;
		this->Wo			= Other.Wo;
		this->Le			= Other.Le;
		this->UV			= Other.UV;
		this->ObjectID		= Other.ObjectID;
		this->LightID		= Other.LightID;

		return *this;
	}

	HOST_DEVICE void GetShader(Shader& Shader)
	{
		switch (this->Type)
		{
			case Enums::Volume:
			{
				switch (gpTracer->RenderSettings.Shading.Type)
				{
					case Enums::BrdfOnly:
					{
						break;
					}

					case Enums::PhaseFunctionOnly:
					{
						break;
					}

					case Enums::Hybrid:
					{
						break;
					}

					case Enums::Modulation:
					{
						break;
					}

					case Enums::Threshold:
					{
						break;
					}

					case Enums::GradientMagnitude:
					{
						break;
					}
				}

				break;
			}

			case Enums::Light:
			{
				break;
			}

			case Enums::Object:
			{
				break;
			}

			case Enums::SlicePlane:
			{
				break;
			}
		}
	}

	Enums::ScatterType	Type;
	bool				Valid;
	float				T;
	Vec3f				P;
	Vec3f				N;
	Vec3f				Wo;
	ColorXYZf			Le;
	Vec2f				UV;
	int					ObjectID;
	int					LightID;
};

}

/*
bool BRDF = false;

float PdfBrdf = 1.0f;

switch (gpTracer->RenderSettings.Shading.Type)
{
	case 0:
	{
		BRDF = true;
		break;
	}

	case 1:
	{
		BRDF = false;
		break;
	}

	
	case 2:
	{
		const float NGM			= GradientMagnitude(SE.P) * gpTracer->Volume.GradientMagnitudeRange.Inv;
		const float Sensitivity	= 25;
		const float ExpGF		= 3;
		const float Exponent	= Sensitivity * powf(gpTracer->RenderSettings.Shading.GradientFactor, ExpGF) * NGM;
		
		PdfBrdf = gpTracer->RenderSettings.Shading.OpacityModulated ? GetOpacity(SE.P) * (1.0f - __expf(-Exponent)) : 1.0f - __expf(-Exponent);
		BRDF = RNG.Get1() <= PdfBrdf;
		break;
	}

	case 3:
	{
		const float NGM = GradientMagnitude(SE.P) * gpTracer->Volume.GradientMagnitudeRange.Inv;
		
		PdfBrdf = 1.0f - powf(1.0f - NGM, 2.0f);
		BRDF = RNG.Get1() < PdfBrdf;
		break;
	}

	case 4:
	{
		const float NGM = GradientMagnitude(SE.P) * gpTracer->Volume.GradientMagnitudeRange.Inv;

		if (NGM > gpTracer->RenderSettings.Shading.GradientThreshold)
			BRDF = true;
		else
			BRDF = false;
	}
	
}

if (BRDF)
	return Shader(Shader::Brdf, SE.N, SE.Wo, GetDiffuse(I), GetSpecular(I), gpTracer->RenderSettings.Shading.IndexOfReflection, GetGlossiness(I));
else
	return Shader(Shader::Phase, SE.N, SE.Wo, GetDiffuse(I), GetSpecular(I), gpTracer->RenderSettings.Shading.IndexOfReflection, GetGlossiness(I));
*/
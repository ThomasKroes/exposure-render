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

#include "Shader.cuh"
#include "RayMarching.cuh"
#include "Lighting.cuh"

DEV ColorXYZf SampleLight(CRNG& RNG, const Vec3f& Pe, Vec3f& Pl, float& Pdf)
{
	const int LID = floorf(RNG.Get1() * gLighting.m_NoLights);

	switch (gLighting.m_Type[LID])
	{
		case 0:
		{
			Pl	= ToVec3f(gLighting.m_P[LID]);
			Pdf	= (Pe - Pl).Length();

			return ColorXYZf(gLighting.m_Color[LID].x, gLighting.m_Color[LID].y, gLighting.m_Color[LID].z);
		}
		
		case 1:
		{
			Pl	= ToVec3f(gLighting.m_P[LID]);
			Pdf	= (Pe - Pl).Length();

			return ColorXYZf(gLighting.m_Color[LID].x, gLighting.m_Color[LID].y, gLighting.m_Color[LID].z);
		}

		case 2:
		{
			Pl	= 150.0f * UniformSampleSphere(RNG.Get2());
			Pdf	= (Pe - Pl).Length();

			return ColorXYZf(gLighting.m_Color[LID].x, gLighting.m_Color[LID].y, gLighting.m_Color[LID].z);
		}
	}
}

DEV ColorXYZf EstimateDirectLight(const CVolumeShader::EType& Type, const float& Density, CLight& Light, CLightingSample& LS, const Vec3f& Wo, const Vec3f& Pe, const Vec3f& N, CRNG& RNG)
{
	ColorXYZf Ld = SPEC_BLACK, Li = SPEC_BLACK, F = SPEC_BLACK;
	
//	CVolumeShader Shader(Type, N, Wo, GetDiffuse(Density), GetSpecular(Density), 2.5f, GetRoughness(Density));
	CVolumeShader Shader(Type, N, Wo, ColorXYZf(1.0f), ColorXYZf(1.0f), 2.5f, 1.0f);

	CRay Rl; 

	float LightPdf = 1.0f, ShaderPdf = 1.0f;

	Vec3f Wi, Pl;

 	Li = SampleLight(RNG, Pe, Pl, LightPdf);
	
	Rl.m_O		= Pe;
	Rl.m_D		= Normalize(Pe - Pl);
	Rl.m_MinT	= 0.0f;
	Rl.m_MaxT	= (Pe - Pl).Length();

	CLight* pLight = NULL;

	Wi = -Rl.m_D; 

	F = Shader.F(Wo, Wi); 

	ShaderPdf = INV_4_PI_F;//.Pdf(Wo, Wi);

	if (!Li.IsBlack() && ShaderPdf > 0.0f && LightPdf > 0.0f && !FreePathRM(Rl, RNG))
	{
		const float WeightMIS = PowerHeuristic(1.0f, LightPdf, 1.0f, ShaderPdf);
		
//		if (Type == CVolumeShader::Brdf)
//			Ld += F * Li * AbsDot(Wi, N) * WeightMIS / LightPdf;

//		if (Type == CVolumeShader::Phase)
//			Ld += F * Li * WeightMIS / LightPdf;
		Ld += F * Li / LightPdf;
	}

	/*
	F = Shader.SampleF(Wo, Wi, ShaderPdf, LS.m_BsdfSample);

	if (!F.IsBlack() && ShaderPdf > 0.0f)
	{
		if (NearestLight(pScene, CRay(Pe, Wi, 0.0f), Li, Pl, pLight, &LightPdf))
		{
			LightPdf = pLight->Pdf(Pe, Wi);

			if (LightPdf > 0.0f && !Li.IsBlack() && !FreePathRM(CRay(Pl, Normalize(Pe - Pl), 0.0f, (Pe - Pl).Length()), RNG)) 
			{
				const float WeightMIS = PowerHeuristic(1.0f, ShaderPdf, 1.0f, LightPdf);

				if (Type == CVolumeShader::Brdf)
					Ld += F * Li * AbsDot(Wi, N) * WeightMIS / ShaderPdf;

				if (Type == CVolumeShader::Phase)
					Ld += F * Li * WeightMIS / ShaderPdf;
			}
		}
	}
	*/

	return Ld;
}

DEV ColorXYZf UniformSampleOneLight(const CVolumeShader::EType& Type, const float& Density, const Vec3f& Wo, const Vec3f& Pe, const Vec3f& N, CRNG& RNG, const bool& Brdf)
{
	/*
	const int NumLights = pScene->m_Lighting.m_NoLights;

 	if (NumLights == 0)
 		return SPEC_BLACK;

	ColorXYZf Li;

	CLightingSample LS;

	LS.LargeStep(RNG);

	const int WhichLight = (int)floorf(LS.m_LightNum * (float)NumLights);

	CLight& Light = pScene->m_Lighting.m_Lights[WhichLight];
	
	return NumLights * EstimateDirectLight(pScene, Type, Density, Light, LS, Wo, Pe, N, RNG);
	*/

	CLight* Light;

	CLightingSample LS;

	LS.LargeStep(RNG);

	return EstimateDirectLight(Type, Density, *Light, LS, Wo, Pe, N, RNG);
}
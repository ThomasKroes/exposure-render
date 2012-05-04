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

#include "raymarching.h"
#include "lights.h"
#include "objects.h"
#include "shader.h"
#include "textures.h"

namespace ExposureRender
{

HOST_DEVICE_NI bool Intersect(const Ray& R, CRNG& RNG)
{
	ScatterEvent SE(Enums::Light);

	if (IntersectsLight(R))
		return true;
	
	if (IntersectsObject(R))
		return true;

	if (ScatterEventInVolume(R, RNG))
		return true;

	return false;
}

HOST_DEVICE_NI bool Visible(const Vec3f& P1, const Vec3f& P2, CRNG& RNG)
{
	if (!gpTracer->RenderSettings.Traversal.Shadows)
		return true;

	Vec3f W = Normalize(P2 - P1);

	const Ray R(P1 + W * RAY_EPS, W, 0.0f, min((P2 - P1).Length() - RAY_EPS_2, gpTracer->RenderSettings.Traversal.MaxShadowDistance));

	return !Intersect(R, RNG);
}

HOST_DEVICE_NI ColorXYZf EstimateDirectLight(const Light& Light, LightingSample& LS, ScatterEvent& SE, CRNG& RNG, Shader& Shader)
{
	Vec3f Wi;
	
	ColorXYZf Li, Ld;

	SurfaceSample SS;

	SampleLight(Light, LS.LightSample, SS, SE, Wi, Li);
	
	ColorXYZf F = Shader.F(SE.Wo, Wi);
	
	float BsdfPdf = Shader.Pdf(SE.Wo, Wi);

	if (!Li.IsBlack() && !F.IsBlack() && BsdfPdf > 0.0f && Visible(SE.P, SS.P, RNG))
	{
		const float LightPdf = DistanceSquared(SE.P, SS.P) / (AbsDot(SS.N, -Wi) * Light.Shape.Area);

		const float Weight = PowerHeuristic(1, LightPdf, 1, BsdfPdf);

		if (Shader.Type == Enums::Brdf)
			Ld += F * Li * (AbsDot(Wi, SE.N) * Weight / LightPdf);
		else
			Ld += F * Li / LightPdf;
	}

	F = Shader.SampleF(SE.Wo, Wi, BsdfPdf, LS.BrdfSample);

	if (F.IsBlack() || BsdfPdf <= 0.0f)
		return Ld;
	
	ScatterEvent SE2(Enums::Light);

	IntersectLights(Ray(SE.P, Wi), SE2);
	
	if (!SE2.Valid || SE2.LightID != Light.ID)
		return Ld;

	Li = SE2.Le;

	if (!Li.IsBlack() && Visible(SE.P, SE2.P, RNG))
	{
		const float LightPdf = DistanceSquared(SE.P, SE2.P) / (AbsDot(SE.N, -Wi) * Light.Shape.Area);

		const float Weight = PowerHeuristic(1, BsdfPdf, 1, LightPdf);

		if (Shader.Type == Enums::Brdf)
			Ld += F * Li * (AbsDot(Wi, SE.N) * Weight / BsdfPdf);
		else
			Ld += F * Li / BsdfPdf;
	}
	
	return Ld;
}

HOST_DEVICE_NI ColorXYZf UniformSampleOneLight(ScatterEvent& SE, CRNG& RNG, LightingSample& LS)
{
	ColorXYZf Ld;

	const float Intensity = GetIntensity(gpTracer->VolumeID, SE.P);

	Ld += gpTracer->Emission1D.Evaluate(Intensity);

	if (gpTracer->LightIDs.Count <= 0)
		return Ld;

	const int LightID = gpTracer->LightIDs[(int)floorf(LS.LightNum * gpTracer->LightIDs.Count)];

	if (LightID < 0)
		return Ld;

	const Light& Light = gpLights[LightID];
	
	Shader Shader;
	
	switch (SE.Type)
	{
		case Enums::Volume:	
			Shader = ExposureRender::Shader(Enums::Brdf, SE.N, SE.Wo, gpTracer->Diffuse1D.Evaluate(Intensity), gpTracer->Specular1D.Evaluate(Intensity), 15.0f, GlossinessExponent(gpTracer->Glossiness1D.Evaluate(Intensity)));
			break;

		case Enums::Object:
		{
			const ColorXYZf Diffuse		= EvaluateTexture(gpObjects[SE.ObjectID].DiffuseTextureID, SE.UV);
			const ColorXYZf Specular	= EvaluateTexture(gpObjects[SE.ObjectID].SpecularTextureID, SE.UV);
			const ColorXYZf Glossiness	= EvaluateTexture(gpObjects[SE.ObjectID].GlossinessTextureID, SE.UV);

			Shader = ExposureRender::Shader(Enums::Brdf, SE.N, SE.Wo, Diffuse, Specular, 15.0f, GlossinessExponent(Glossiness.Y()));
			break;
		}
	}
	/**/

	Ld += EstimateDirectLight(Light, LS, SE, RNG, Shader);

	return (float)gpTracer->LightIDs.Count * Ld;
}

}
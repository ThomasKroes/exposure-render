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

	return Ld;

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
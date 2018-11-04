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

#include "geometry.h"
#include "volumes.h"
#include "transferfunction.h"
#include "shapes.h"
#include "scatterevent.h"

namespace ExposureRender
{

HOST_DEVICE_NI void SampleVolume(Ray R, CRNG& RNG, ScatterEvent& SE)
{
	float MinT;
	float MaxT;
	
	Intersection Int;

	IntersectBox(R, gpVolumes[gpTracer->VolumeID].BoundingBox.MinP, gpVolumes[gpTracer->VolumeID].BoundingBox.MaxP, Int);

	if (!Int.Valid)
		return;

	MinT = max(Int.NearT, R.MinT);
	MaxT = min(Int.FarT, R.MaxT);

	const float S	= -log(RNG.Get1()) / gpTracer->RenderSettings.Shading.DensityScale;
	float Sum		= 0.0f;
	float SigmaT	= 0.0f;

	Vec3f Ps;

	const float StepSize = gpTracer->RenderSettings.Traversal.StepFactorPrimary * gpVolumes[gpTracer->VolumeID].MinStep;

	MinT += RNG.Get1() * StepSize;

	while (Sum < S)
	{
		Ps = R.O + MinT * R.D;

		if (MinT >= MaxT)
			return;
		
		float Intensity = GetIntensity(gpTracer->VolumeID, Ps);

		SigmaT	= gpTracer->RenderSettings.Shading.DensityScale * gpTracer->Opacity1D.Evaluate(Intensity);

		Sum		+= SigmaT * StepSize;
		MinT	+= StepSize;
	}

	SE.SetValid(MinT, Ps, NormalizedGradient(gpTracer->VolumeID, Ps), -R.D, ColorXYZf());
}

HOST_DEVICE_NI bool ScatterEventInVolume(Ray R, CRNG& RNG)
{
	float MinT;
	float MaxT;
	Vec3f Ps;

	Intersection Int;
		
	IntersectBox(R, gpVolumes[gpTracer->VolumeID].BoundingBox.MinP, gpVolumes[gpTracer->VolumeID].BoundingBox.MaxP, Int);
	
	if (!Int.Valid)
		return false;

	MinT = max(Int.NearT, R.MinT);
	MaxT = min(Int.FarT, R.MaxT);

	const float S	= -log(RNG.Get1()) / gpTracer->RenderSettings.Shading.DensityScale;
	float Sum		= 0.0f;
	float SigmaT	= 0.0f;

	const float StepSize = gpTracer->RenderSettings.Traversal.StepFactorShadow * gpVolumes[gpTracer->VolumeID].MinStep;

	MinT += RNG.Get1() * StepSize;

	while (Sum < S)
	{
		Ps = R.O + MinT * R.D;

		if (MinT > MaxT)
			return false;
		
		float Intensity = GetIntensity(gpTracer->VolumeID, Ps);

		SigmaT	= gpTracer->RenderSettings.Shading.DensityScale * gpTracer->Opacity1D.Evaluate(Intensity);

		Sum		+= SigmaT * StepSize;
		MinT	+= StepSize;
	}

	return true;
}

/*
struct Photon{
  Vec3f origin;
  Vec3f direction;
  float energy;

  // gamma photon
  float photonEnergy;
  float sigma;
};

#define EPS (0.000001f)

__device__
float sign(float num){
  if(num<0.0f) return(-1.0f);
  if(num>0.0f) return(1.0f);
  return(0.0f);
}


HOST_DEVICE void lightShootDDAWoodcock(const Ray& R, CRNG& RNG, ScatterEvent& SE)
{
	Intersection Int;

	IntersectBox(R, ToVec3f(gVolumeProperties.MinAABB), ToVec3f(gVolumeProperties.MaxAABB), Int);

	if (!Int.Valid)
		return;
	else
	{
//		SE.SetValid((R(Int.NearT) - R.O).Length(), R(Int.NearT), NormalizedGradient(R(Int.NearT)), -R.D, ColorXYZf());
//		return;
	}

	Photon photon;

	photon.origin = R(Int.NearT + 0.000001f);
	photon.direction = R.D;

	float c_macrocellSize = 1.0f / 8.0f;

  Vec3f cellIndex;
  cellIndex[0] = floor(photon.origin[0] / c_macrocellSize);
  cellIndex[1] = floor(photon.origin[1] / c_macrocellSize);
  cellIndex[2] = floor(photon.origin[2] / c_macrocellSize);
  
  Vec3f t(0.0f);

  if(photon.direction[0] > EPS){
    t[0] = ((cellIndex[0] + 1) * c_macrocellSize - photon.origin[0]) / photon.direction[0];
  } else {
    if(photon.direction[0] < -EPS){
      t[0] = (cellIndex[0] * c_macrocellSize - photon.origin[0]) / photon.direction[0];
    } else {
      t[0] = 1000.0f;
    }
  }
  if(photon.direction[1] > EPS){
    t[1] = ((cellIndex[1] + 1) * c_macrocellSize - photon.origin[1]) / photon.direction[1];
  } else {
    if(photon.direction[1] < -EPS){
      t[1] = (cellIndex[1] * c_macrocellSize - photon.origin[1]) / photon.direction[1];
    } else {
      t[1] = 1000.0f;
    }
  }
  if(photon.direction[2] > EPS){
    t[2] = ((cellIndex[2] + 1) * c_macrocellSize - photon.origin[2]) / photon.direction[2];
  } else {
    if(photon.direction[2] < -EPS){
      t[2] = (cellIndex[2] * c_macrocellSize - photon.origin[2]) / photon.direction[2];
    } else {
      t[2] = 1000.0f;
    }
  }

  Vec3f cpv;
  cpv[0] = c_macrocellSize / fabs(photon.direction[0]);
  cpv[1] = c_macrocellSize / fabs(photon.direction[1]);
  cpv[2] = c_macrocellSize / fabs(photon.direction[2]);

  Vec3f samplePos = photon.origin;

  int steps = 0;
  
  float c_densityScale = gVolumeProperties.DensityScale;

  bool virtualHit = true;
  while(virtualHit) {
    float sigmaMax = tex3D(gTexExtinction, photon.origin[0], photon.origin[1], photon.origin[2]);
    float lastSigmaMax = sigmaMax;
    float ds = min(t[0], min(t[1], t[2]));
    float sigmaSum = sigmaMax * ds;
	float s = -log(1.0f - RNG.Get1()) / c_densityScale;
    float tt = min(t[0], min(t[1], t[2]));
    Vec3f entry;
    Vec3f exit = photon.origin + tt * photon.direction;

    while(sigmaSum < s){
      if(steps++ > 100.0f){
        photon.energy = 0.0f;
        return;
      }
      entry = exit;

      if (entry[0] <= -0.5f || entry[0] >= 0.5f ||
          entry[1] <= -0.5f || entry[1] >= 0.5f ||
          entry[2] <= -0.5f || entry[2] >= 0.5f){
        photon.energy = 0.0f;
        return;
      }

      if(t[0]<t[1] && t[0]<t[2]){
        cellIndex[0] += sign(photon.direction[0]);
        t[0] += cpv[0];
      } else {
        if(t[1]<t[0] && t[1]<t[2]){
          cellIndex[1] += sign(photon.direction[1]);
          t[1] += cpv[1];
        } else {
          cellIndex[2] += sign(photon.direction[2]);
          t[2] += cpv[2];
        }
      }

      tt = min(t[0], min(t[1], t[2]));
      exit = photon.origin + tt * photon.direction;
      ds = (exit-entry).Length();
      sigmaSum += ds * sigmaMax;
      lastSigmaMax = sigmaMax;
      Vec3f ePos = (exit + entry) / 2.0f;
      sigmaMax = tex3D(gTexExtinction, ePos[0], ePos[1], ePos[2]);
      samplePos = entry;
    }

    float cS = (s - (sigmaSum - ds * lastSigmaMax)) / lastSigmaMax;
    samplePos += photon.direction * cS;

    if (photon.origin[0] <= -0.5f || photon.origin[0] >= 0.5f ||
        photon.origin[1] <= -0.5f || photon.origin[1] >= 0.5f ||
        photon.origin[2] <= -0.5f || photon.origin[2] >= 0.5f){
      photon.energy = 0.0f;
      return;
    }

	if(tex3D(gTexIntensity, samplePos[0], samplePos[1], samplePos[2]) / tex3D(gTexExtinction, samplePos[0], samplePos[1], samplePos[2]) > RNG.Get1()){
      virtualHit = false;
    } else {
      photon.origin = exit;
    }
  }

//  photon.energy = photon.energy * c_albedo;
//  photon.origin = samplePos;

	SE.SetValid((samplePos - R.O).Length(), samplePos, NormalizedGradient(samplePos), -R.D, ColorXYZf());
}
*/

}
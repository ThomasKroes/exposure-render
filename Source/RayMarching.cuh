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

#include "Geometry.cuh"
#include "Volume.cuh"

namespace ExposureRender
{

DEVICE_NI void SampleVolume(Ray R, CRNG& RNG, ScatterEvent& SE)
{
	float MinT;
	float MaxT;
	
	Intersection Int;

	IntersectBox(R, gpVolumes->List[gpTracer->VolumeIDs[0]].MinAABB, gpVolumes->List[gpTracer->VolumeIDs[0]].MaxAABB, Int);

	if (!Int.Valid)
		return;

	MinT = max(Int.NearT, R.MinT);
	MaxT = min(Int.FarT, R.MaxT);

	const float S	= -log(RNG.Get1()) / gpTracer->RenderSettings.Shading.DensityScale;
	float Sum		= 0.0f;
	float SigmaT	= 0.0f;

	Vec3f Ps;

	MinT += RNG.Get1() * gpTracer->RenderSettings.Traversal.StepSize;

	while (Sum < S)
	{
		Ps = R.O + MinT * R.D;

		if (MinT >= MaxT)
			return;
		
		SigmaT	= gpTracer->RenderSettings.Shading.DensityScale * GetOpacity(Ps);

		Sum			+= SigmaT * gpTracer->RenderSettings.Traversal.StepSize;
		MinT	+= gpTracer->RenderSettings.Traversal.StepSize;
	}

	SE.SetValid(MinT, Ps, NormalizedGradient(Ps), -R.D, ColorXYZf());
}

// Determines if there is a scatter event along the ray
DEVICE_NI bool ScatterEventInVolume(Ray R, CRNG& RNG)
{
	float MinT;
	float MaxT;
	Vec3f Ps;

	Intersection Int;
		
	IntersectBox(R, gpVolumes->List[gpTracer->VolumeIDs[0]].MinAABB, gpVolumes->List[gpTracer->VolumeIDs[0]].MaxAABB, Int);
	
	if (!Int.Valid)
		return false;

	MinT = max(Int.NearT, R.MinT);
	MaxT = min(Int.FarT, R.MaxT);

	const float S	= -log(RNG.Get1()) / gpTracer->RenderSettings.Shading.DensityScale;
	float Sum		= 0.0f;
	float SigmaT	= 0.0f;

	MinT += RNG.Get1() * gpTracer->RenderSettings.Traversal.StepSizeShadow;

	while (Sum < S)
	{
		Ps = R.O + MinT * R.D;

		if (MinT > MaxT)
			return false;
		
		SigmaT	= gpTracer->RenderSettings.Shading.DensityScale * GetOpacity(Ps);

		Sum			+= SigmaT * gpTracer->RenderSettings.Traversal.StepSizeShadow;
		MinT	+= gpTracer->RenderSettings.Traversal.StepSizeShadow;
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


DEVICE_NI void lightShootDDAWoodcock(const Ray& R, CRNG& RNG, ScatterEvent& SE)
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
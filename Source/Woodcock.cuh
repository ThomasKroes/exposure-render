/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "Geometry.h"
#include "Scene.h"

#include "Utilities.cuh"

#define EPS (0.000001f)

DEV float sign(float num)
{
  if (num<0.0f)
	  return -1.0f;

  if (num > 0.0f)
	  return 1.0f;

  return 0.0f;
}

struct Photon{
  float3 origin;
  float3 direction;
  float energy;

  // gamma photon
  float photonEnergy;
  float sigma;
};







DEV inline bool SampleDistanceDdaWoodcock(CRay R, CRNG& RNG, Vec3f& P, CScene* pScene, int Component)
{
	float MinT = 0.0f, MaxT = 0.0f;

 	if (!pScene->m_BoundingBox.Intersect(R, &MinT, &MaxT))
 		return false;
	
	Photon photon;

	photon.origin = make_float3(R.m_O.x, R.m_O.y, R.m_O.z);
	photon.direction = normalize(make_float3(R.m_D.x, R.m_D.y, R.m_D.z));
	
	photon.origin = photon.origin + photon.direction * (MinT + RNG.Get1() * 0.001f);

	Vec3f c_macrocellSize(1.0f / (float)pScene->m_ExtinctionSize.width);//, 1.0f / (float)pScene->m_ExtinctionSize.height, 1.0f / (float)pScene->m_ExtinctionSize.depth);
//	Vec3f c_macrocellSize(1.0f / 16.0f, 1.0f / 16.0f, 1.0f / 16.0f);

	float3 cellIndex;
  cellIndex.x = floor(photon.origin.x / c_macrocellSize.x);
  cellIndex.y = floor(photon.origin.y / c_macrocellSize.y);
  cellIndex.z = floor(photon.origin.z / c_macrocellSize.z);
  float3 t = make_float3(0.0f);

	if(photon.direction.x > EPS){
    t.x = ((cellIndex.x + 1) * c_macrocellSize.x - photon.origin.x) / photon.direction.x;
  } else {
    if(photon.direction.x < -EPS){
      t.x = (cellIndex.x * c_macrocellSize.x - photon.origin.x) / photon.direction.x;
    } else {
      t.x = 1000.0f;
    }
  }
  if(photon.direction.y > EPS){
    t.y = ((cellIndex.y + 1) * c_macrocellSize.y - photon.origin.y) / photon.direction.y;
  } else {
    if(photon.direction.y < -EPS){
      t.y = (cellIndex.y * c_macrocellSize.y - photon.origin.y) / photon.direction.y;
    } else {
      t.y = 1000.0f;
    }
  }
  if(photon.direction.z > EPS){
    t.z = ((cellIndex.z + 1) * c_macrocellSize.z - photon.origin.z) / photon.direction.z;
  } else {
    if(photon.direction.z < -EPS){
      t.z = (cellIndex.z * c_macrocellSize.z - photon.origin.z) / photon.direction.z;
    } else {
      t.z = 1000.0f;
    }
  }

  float3 cpv;
  cpv.x = c_macrocellSize.x / fabs(photon.direction.x);
  cpv.y = c_macrocellSize.y / fabs(photon.direction.y);
  cpv.z = c_macrocellSize.z / fabs(photon.direction.z);

  float3 samplePos = photon.origin;

  int steps = 0;
  
 bool virtualHit = true;
  while(virtualHit) {
    float sigmaMax = tex3D(gTexExtinction, photon.origin.x, photon.origin.y, photon.origin.z) / 255.0f;//tex1D(texTt, photon.photonEnergy);
    float lastSigmaMax = sigmaMax;
    float ds = min(t.x, min(t.y, t.z));
    float sigmaSum = sigmaMax * ds;
    float s = -log(1.0f - RNG.Get1()) / (pScene->m_DensityScale * 100.0f);
    float tt = min(t.x, min(t.y, t.z));
    float3 entry;
    float3 exit = photon.origin + tt * photon.direction;

    while(sigmaSum < s){
      if(steps++ > 100.0f){
        photon.energy = 0.0f;
        return false;
      }
      entry = exit;

      if (entry.x <= 0.0f || entry.x >= 1.0f ||
          entry.y <= 0.0f || entry.y >= 1.0f ||
          entry.z <= 0.0f || entry.z >= 1.0f){
        photon.energy = 0.0f;
        return false;
      }

      if(t.x<t.y && t.x<t.z){
        cellIndex.x += sign(photon.direction.x);
        t.x += cpv.x;
      } else {
        if(t.y<t.x && t.y<t.z){
          cellIndex.y += sign(photon.direction.y);
          t.y += cpv.y;
        } else {
          cellIndex.z += sign(photon.direction.z);
          t.z += cpv.z;
        }
      }

      tt = min(t.x, min(t.y, t.z));
      exit = photon.origin + tt * photon.direction;
      ds = length(exit-entry);
      sigmaSum += ds * sigmaMax;
      lastSigmaMax = sigmaMax;
      float3 ePos = (exit + entry) / 2.0f;
      sigmaMax = tex3D(gTexExtinction, ePos.x, ePos.y, ePos.z) / 255.0f;//tex1D(texTt, photon.photonEnergy);
      samplePos = entry;
    }

    float cS = (s - (sigmaSum - ds * lastSigmaMax)) / lastSigmaMax;
    samplePos += photon.direction * cS;

    if (photon.origin.x <= 0.0f || photon.origin.x >= 1.0f ||
        photon.origin.y <= 0.0f || photon.origin.y >= 1.0f ||
        photon.origin.z <= 0.0f || photon.origin.z >= 1.0f){
      photon.energy = 0.0f;
      return false;
    }

    if((tex3D(gTexDensity, samplePos.x, samplePos.y, samplePos.z) / 255.0f) / (tex3D(gTexExtinction, samplePos.x, samplePos.y, samplePos.z) / 255.0f) > RNG.Get1()){
      virtualHit = false;
    } else {
      photon.origin = samplePos;
    }
  }

//    if (virtualHit)
//  	  return false;

//   photon.energy = photon.energy * c_albedo;
//   photon.origin = samplePos;

  P.x = samplePos.x;/// + RNG.Get1() * 0.01f;
  P.y = samplePos.y;// + RNG.Get1() * 0.01f;
  P.z = samplePos.z;// + RNG.Get1() * 0.01f;

   return true;
}
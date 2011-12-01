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

#include "Geometry.h"

#include "Utilities.cuh"

#define EPS (0.000001f)

__device__
float sign(float num){
  if(num<0.0f) return(-1.0f);
  if(num>0.0f) return(1.0f);
  return(0.0f);
}

__device__ bool lightShootDDAWoodcock(CRay& R, CRNG& RNG, Vec3f& P)
{
	const int TID = threadIdx.y * blockDim.x + threadIdx.x;

	__shared__ float MinT[KRNL_SINGLE_SCATTERING_BLOCK_SIZE];
	__shared__ float MaxT[KRNL_SINGLE_SCATTERING_BLOCK_SIZE];

	if (!IntersectBox(R, &MinT[TID], &MaxT[TID]))
		return false;
	else
		return true;

	MinT[TID] = max(MinT[TID], R.m_MinT);
	MaxT[TID] = min(MaxT[TID], R.m_MaxT);

	Vec3f pos, dir;

	pos = R(MinT[TID]);
	dir = R.m_D;

	float3 cellIndex;

	cellIndex.x = floor(pos.x / gVolumeInfo.m_MacroCellSize);
	cellIndex.y = floor(pos.y / gVolumeInfo.m_MacroCellSize);
	cellIndex.z = floor(pos.z / gVolumeInfo.m_MacroCellSize);
	
	float3 t = make_float3(0.0f);

	if (dir.x > EPS)
	{
		t.x = ((cellIndex.x + 1) * gVolumeInfo.m_MacroCellSize - pos.x) / dir.x;
	}
	else
	{
		if (dir.x < -EPS)
		{
			t.x = (cellIndex.x * gVolumeInfo.m_MacroCellSize - pos.x) / dir.x;
		}
		else
		{
			t.x = 1000.0f;
		}
	}

	if (dir.y > EPS)
	{
		t.y = ((cellIndex.y + 1) * gVolumeInfo.m_MacroCellSize - pos.y) / dir.y;
	}
	else
	{
		if (dir.y < -EPS)
		{
			t.y = (cellIndex.y * gVolumeInfo.m_MacroCellSize - pos.y) / dir.y;
		}
		else
		{
			t.y = 1000.0f;
		}
	}
	
	if (dir.z > EPS)
	{
		t.z = ((cellIndex.z + 1) * gVolumeInfo.m_MacroCellSize - pos.z) / dir.z;
	}
	else
	{
		if (dir.z < -EPS)
		{
			t.z = (cellIndex.z * gVolumeInfo.m_MacroCellSize - pos.z) / dir.z;
		}
		else
		{
			t.z = 1000.0f;
		}
	}

	float3 cpv;
	cpv.x = gVolumeInfo.m_MacroCellSize / fabs(dir.x);
	cpv.y = gVolumeInfo.m_MacroCellSize / fabs(dir.y);
	cpv.z = gVolumeInfo.m_MacroCellSize / fabs(dir.z);

  float3 samplePos = FromVec3f(pos);

  int steps = 0;
  
  bool virtualHit = true;
  while(virtualHit) {
    float sigmaMax = tex3D(gTexExtinction, pos.x, pos.y, pos.z);
    float lastSigmaMax = sigmaMax;
    float ds = min(t.x, min(t.y, t.z));
    float sigmaSum = sigmaMax * ds;
	float s = -log(1.0f - RNG.Get1()) / gVolumeInfo.m_DensityScale;
    float tt = min(t.x, min(t.y, t.z));
    float3 entry;
    float3 exit = FromVec3f(pos) + FromVec3f(tt * dir);

    while(sigmaSum < s){
      if(steps++ > 100.0f){
 //       photon->energy = 0.0f;
        return false;
      }
      entry = exit;

	  /*
      if (entry.x <= 0.0f || entry.x >= 1.0f ||
          entry.y <= 0.0f || entry.y >= 1.0f ||
          entry.z <= 0.0f || entry.z >= 1.0f){
  //      photon->energy = 0.0f;
        return  false;
      }
	  */

      if(t.x<t.y && t.x<t.z){
        cellIndex.x += sign(dir.x);
        t.x += cpv.x;
      } else {
        if(t.y<t.x && t.y<t.z){
          cellIndex.y += sign(dir.y);
          t.y += cpv.y;
        } else {
          cellIndex.z += sign(dir.z);
          t.z += cpv.z;
        }
      }

      tt = min(t.x, min(t.y, t.z));
      exit = FromVec3f(pos) + FromVec3f(tt * dir);
      ds = length(exit-entry);
      sigmaSum += ds * sigmaMax;
      lastSigmaMax = sigmaMax;
      float3 ePos = (exit + entry) / 2.0f;
      sigmaMax = tex3D(gTexExtinction, ePos.x, ePos.y, ePos.z);
      samplePos = entry;
    }

    float cS = (s - (sigmaSum - ds * lastSigmaMax)) / lastSigmaMax;
    samplePos += FromVec3f(dir * cS);

	/*
    if (pos.x <= 0.0f || pos.x >= 1.0f ||
        pos.y <= 0.0f || pos.y >= 1.0f ||
        pos.z <= 0.0f || pos.z >= 1.0f){
//      photon->energy = 0.0f;
      return false;
    }
	*/

	if(tex3D(gTexDensity, samplePos.x, samplePos.y, samplePos.z) / tex3D(gTexExtinction, samplePos.x, samplePos.y, samplePos.z) > RNG.Get1()){
      virtualHit = false;
    } else {
      pos = ToVec3f(exit);
    }
  }

//  photon->energy = photon->energy * c_albedo;
  R.m_O = ToVec3f(samplePos);
  P = R.m_O;

  return true;
}
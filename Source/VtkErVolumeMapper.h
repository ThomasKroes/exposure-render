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

#include "vtkErBoxWidget.h"

#include <vtkVolumeMapper.h>
#include <vtkSmartPointer.h>
#include <vtkVolumeProperty.h>
#include <vtkCommand.h>
#include <vtkLightCollection.h>
#include <vtkCamera.h>
#include <vtkTextActor.h>

#include "General.cuh"

class vtkErVolumeMapper;
class vtkErLight;
class vtkErVolumeProperty;

class vtkErResetCommand : public vtkCommand
{
public:
	static vtkErResetCommand* New() { return new vtkErResetCommand; };

	virtual void Execute(vtkObject*, unsigned long, void *);
	
	void SetVolumeMapper(vtkErVolumeMapper* pVolumeMapper);

protected:
	vtkErResetCommand() { this->VolumeMapper = NULL; };
	~vtkErResetCommand() {};

	vtkErVolumeMapper*	VolumeMapper;
};

class vtkErUpdateSlicingCommand : public vtkCommand
{
public:
	static vtkErUpdateSlicingCommand* New() { return new vtkErUpdateSlicingCommand; };

	virtual void Execute(vtkObject*, unsigned long, void *);
	
	void SetVolumeMapper(vtkErVolumeMapper* pVolumeMapper);

protected:
	vtkErUpdateSlicingCommand() { this->VolumeMapper = NULL; };
	~vtkErUpdateSlicingCommand() {};

	vtkErVolumeMapper*	VolumeMapper;
};

class vtkErUpdateLightingCommand : public vtkCommand
{
public:
	static vtkErUpdateLightingCommand* New() { return new vtkErUpdateLightingCommand; };

	virtual void Execute(vtkObject*, unsigned long, void *);
	
	void SetVolumeMapper(vtkErVolumeMapper* pVolumeMapper);

protected:
	vtkErUpdateLightingCommand() { this->VolumeMapper = NULL; };
	~vtkErUpdateLightingCommand() {};

	vtkErVolumeMapper*	VolumeMapper;
};

class vtkErUpdateCameraCommand : public vtkCommand
{
public:
	static vtkErUpdateCameraCommand* New() { return new vtkErUpdateCameraCommand; };

	virtual void Execute(vtkObject*, unsigned long, void *);
	
	void SetVolumeMapper(vtkErVolumeMapper* pVolumeMapper);

protected:
	vtkErUpdateCameraCommand() { this->VolumeMapper = NULL; };
	~vtkErUpdateCameraCommand() {};

	vtkErVolumeMapper*	VolumeMapper;
};

class vtkErUpdateBlurCommand : public vtkCommand
{
public:
	static vtkErUpdateBlurCommand* New() { return new vtkErUpdateBlurCommand; };

	virtual void Execute(vtkObject*, unsigned long, void *);
	
	void SetVolumeMapper(vtkErVolumeMapper* pVolumeMapper);

protected:
	vtkErUpdateBlurCommand() { this->VolumeMapper = NULL; };
	~vtkErUpdateBlurCommand() {};

	vtkErVolumeMapper*	VolumeMapper;
};

class VTK_ER_CORE_EXPORT vtkErNoiseReduction : public vtkObject
{
public:
	static vtkErNoiseReduction *New();

	void Reset();

	vtkGetMacro(Enabled, float);
	vtkSetMacro(Enabled, float);

	vtkGetMacro(WindowRadius, float);
	vtkSetMacro(WindowRadius, float);

	vtkGetMacro(Noise, float);
	vtkSetMacro(Noise, float);

	vtkGetMacro(WeightThreshold, float);
	vtkSetMacro(WeightThreshold, float);

	vtkGetMacro(LerpThreshold, float);
	vtkSetMacro(LerpThreshold, float);

	vtkGetMacro(LerpC, float);
	vtkSetMacro(LerpC, float);

protected:
	vtkErNoiseReduction() {};
	virtual ~vtkErNoiseReduction() {};

public:
	float	Enabled;
	float	WindowRadius;
	float	WindowArea;
	float	InvWindowArea;
	float	Noise;
	float	WeightThreshold;
	float	LerpThreshold;
	float	LerpC;
};



class VTK_ER_CORE_EXPORT vtkErVolumeMapper : public vtkVolumeMapper
{
public:
	vtkTypeMacro(vtkErVolumeMapper, vtkVolumeMapper);
    static vtkErVolumeMapper* New();

	vtkErVolumeMapper operator=(const vtkErVolumeMapper&);
    vtkErVolumeMapper(const vtkErVolumeMapper&);
    
	virtual void SetInput(vtkImageData* pImageData);
    
	virtual void Render(vtkRenderer* pRenderer, vtkVolume* pVolume);

	virtual int FillInputPortInformation(int, vtkInformation*);

   vtkImageData* GetOutput() { return NULL;  }

	void PrintSelf(ostream& os, vtkIndent indent);

	vtkErVolumeMapper();
    virtual ~vtkErVolumeMapper();

	void UploadVolumeProperty(vtkVolumeProperty* pVolumeProperty);

	unsigned int TextureID;

	vtkGetMacro(SliceWidget, vtkErBoxWidget*);
	void SetSliceWidget(vtkErBoxWidget* pSliceWidget);

	void Reset();

	vtkGetMacro(MacroCellSize, int);
	vtkSetMacro(MacroCellSize, int);

	vtkGetMacro(ShowFPS, bool);
	vtkSetMacro(ShowFPS, bool);

	vtkGetMacro(SceneScale, double);
	vtkSetMacro(SceneScale, double);

	void AddLight(vtkLight* pLight);
	void RemoveLight(vtkLight* pLight);

	vtkErNoiseReduction* GetNoiseReduction();

protected:
	vtkRenderer*									Renderer;
	vtkCamera*										ActiveCamera;
	vtkVolumeProperty*								VolumeProperty;

	vtkErBoxWidget*									SliceWidget;

	vtkSmartPointer<vtkErResetCommand>				ResetCallBack;
	vtkSmartPointer<vtkErUpdateSlicingCommand>		UpdateSlicingCommand;
	vtkSmartPointer<vtkErUpdateLightingCommand>		UpdateLightingCommand;
	vtkSmartPointer<vtkErUpdateCameraCommand>		UpdateCameraCommand;
	vtkSmartPointer<vtkErUpdateBlurCommand>			UpdateBlur;

	int												MacroCellSize;
	bool											ShowFPS;

	double											SceneScale;

	vtkSmartPointer<vtkLightCollection>				Lights;

	vtkLightCollection* GetLights();

	vtkSmartPointer<vtkErNoiseReduction>			NoiseReduction;

	//BTX
	Volume											Volume;
	Camera											Camera;
	Lighting										Lighting;
	Slicing											Slicing;
	Denoise											Denoise;
	Scattering										Scattering;
	Blur											Blur;
	FrameBuffer										FrameBuffer;
	//ETX

	CHostBuffer2D<ColorRGBAuc>						Host;

	vtkImageData*									Intensity;
	vtkImageData*									GradientMagnitude;

	vtkSmartPointer<vtkTextActor>					BorderWidget;

	friend class vtkErUpdateSlicingCommand;
	friend class vtkErUpdateLightingCommand;
	friend class vtkErUpdateCameraCommand;
	friend class vtkErUpdateBlurCommand;
};

// http://www.na-mic.org/svn/Slicer3/branches/cuda/Modules/VolumeRenderingCuda/
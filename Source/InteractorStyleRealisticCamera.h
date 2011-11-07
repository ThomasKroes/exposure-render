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

// VTK
#include <vtkInteractorStyleUser.h>
#include <vtkObjectFactory.h>

#include "Flags.h"
#include "Camera.h"

// Define interaction style for controlling a realistic camera
class vtkRealisticCameraStyle : public vtkInteractorStyleUser
{
public:
	enum EMouseButtonFlag
	{
		Left	= 0x0001,
		Middle	= 0x0002,
		Right	= 0x0004
	};

	static vtkRealisticCameraStyle* New();
	vtkTypeMacro(vtkRealisticCameraStyle, vtkInteractorStyleUser);

	virtual void OnLeftButtonDown(void);
	virtual void OnLeftButtonUp(void);
	virtual void OnRightButtonDown(void);
	virtual void OnRightButtonUp(void);
	virtual void OnMiddleButtonDown(void);
	virtual void OnMiddleButtonUp(void);
	virtual void OnMouseWheelForward(void);
	virtual void OnMouseWheelBackward(void);
	virtual void OnMouseMove(void);

	int m_OldPos[2];
	int m_NewPos[2];

	// Camera sensitivity to mouse movement
	static float m_OrbitSpeed;			
	static float m_PanSpeed;
	static float m_ZoomSpeed;
	static float m_ContinuousZoomSpeed;
	static float m_ApertureSpeed;
	static float m_FovSpeed;
};
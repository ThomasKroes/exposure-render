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

#include "vtkErAreaLight.h"

class vtkActor;
class vtkPolyDataMapper;
class vtkPoints;
class vtkPolyData;
class vtkProp;
class vtkProperty;
class vtkSphereSource;
class vtkCellPicker;
class vtkErPointWidget;
class vtkPWCallback;
class vtkPW1Callback;
class vtkPW2Callback;

class EXPOSURE_RENDER_DLL vtkErAreaLightWidget : public vtk3DWidget
{
	vtkTypeMacro(vtkErAreaLightWidget, vtk3DWidget);
	static vtkErAreaLightWidget *New();

public:
	virtual void SetEnabled(int Enabled);
	virtual void PlaceWidget(double Bounds[6]);
	void PlaceWidget(void) { this->Superclass::PlaceWidget(); }

	vtkGetMacro(AreaLight, vtkErAreaLight*);
	vtkSetMacro(AreaLight, vtkErAreaLight*);

	void CreateDefaultProperties(void);

	void SetPosition(double x, double y, double z);
	void SetPosition(double x[3]) { this->SetPosition(x[0], x[1], x[2]); }
	double* GetPosition(void) { return this->AreaLight->GetPosition(); }
	void GetPosition(double xyz[3]) { this->AreaLight->GetPosition(xyz); }

	void SetTarget(double x, double y, double z);
	void SetTarget(double x[3]) { this->SetTarget(x[0], x[1], x[2]); }
	double* GetTarget(void) { return this->AreaLight->GetFocalPoint(); }
	void GetTarget(double xyz[3]) { this->AreaLight->GetFocalPoint(xyz); }

protected:
	//BTX
	vtkErAreaLightWidget();
	virtual ~vtkErAreaLightWidget();

	friend class vtkPWCallback;

	vtkErAreaLight*	AreaLight;

	vtkSmartPointer<vtkActor>			LineActor;
	vtkSmartPointer<vtkPolyDataMapper>	LineMapper;
	vtkSmartPointer<vtkLineSource>		LineSource;

	vtkSmartPointer<vtkActor>			PositionHandle;
	vtkSmartPointer<vtkPolyDataMapper>	PositionHandleMapper;
	vtkSmartPointer<vtkSphereSource>	PositionHandleGeometry;

	vtkSmartPointer<vtkActor>			TargetHandle;
	vtkSmartPointer<vtkPolyDataMapper>	TargetHandleMapper;
	vtkSmartPointer<vtkSphereSource>	TargetHandleGeometry;

	vtkSmartPointer<vtkCellPicker>		HandlePicker;
	vtkSmartPointer<vtkCellPicker>		LinePicker;
	
	vtkActor*							CurrentHandle;

	vtkSmartPointer<vtkProperty>		HandleProperty;
	vtkSmartPointer<vtkProperty>		SelectedHandleProperty;
	vtkSmartPointer<vtkProperty>		LineProperty;
	vtkSmartPointer<vtkProperty>		SelectedLineProperty;

	vtkSmartPointer<vtkErPointWidget>	PointWidget;
	vtkSmartPointer<vtkErPointWidget>	PointWidget1;
	vtkSmartPointer<vtkErPointWidget>	PointWidget2;
	vtkSmartPointer<vtkPWCallback>		PWCallback;
	vtkSmartPointer<vtkPW1Callback>		PW1Callback;
	vtkSmartPointer<vtkPW2Callback>		PW2Callback;
	vtkErPointWidget*					CurrentPointWidget;

	void BuildRepresentation(void);

	static void ProcessEvents(vtkObject* pObject, unsigned long event, void* pClientdata, void* pCalldata);

	int State;
	
	enum WidgetState
	{
		Start = 0,
		MovingHandle,
		MovingLine,
		Scaling,
		Outside
	};

	void OnLeftButtonDown();
	void OnLeftButtonUp();
	void OnMiddleButtonDown();
	void OnMiddleButtonUp();
	void OnRightButtonDown();
	void OnRightButtonUp();
	virtual void OnMouseMove();

	virtual void SizeHandles(void);
	int HighlightHandle(vtkProp* pProp);
	void HighlightHandles(int Highlight);

	void EnablePointWidget(void);
	void DisablePointWidget(void);
	int ForwardEvent(unsigned long event);

	double LastPosition[3];

	void HighlightLine(int Highlight);
	//ETX
};
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

#include "vtkErCoreDll.h"

#include "vtkPolyDataSourceWidget.h"

#include "vtkActor.h"
#include "vtkAssemblyNode.h"
#include "vtkAssemblyPath.h"
#include "vtkCallbackCommand.h"
#include "vtkCamera.h"
#include "vtkCellArray.h"
#include "vtkCellPicker.h"
#include "vtkConeSource.h"
#include "vtkDoubleArray.h"
#include "vtkFloatArray.h"
#include "vtkLineSource.h"
#include "vtkMath.h"
#include "vtkObjectFactory.h"
#include "vtkPlane.h"
#include "vtkPlaneSource.h"
#include "vtkPlanes.h"
#include "vtkPolyData.h"
#include "vtkPolyDataMapper.h"
#include "vtkProperty.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkRenderer.h"
#include "vtkSphereSource.h"
#include "vtkTransform.h"
#include "vtkVolume.h"
#include "vtkPoints.h"
#include "vtkCubeSource.h"
#include "vtkLabeledDataMapper.h"
#include "vtkPointSource.h"
#include "vtkActor2D.h"
#include "vtkProperty2D.h"
#include "vtkErSlicePlane.h"
#include "vtkSmartPointer.h"

#include "vtkErSlicePlane.h"

class VTK_ER_CORE_EXPORT vtkErSliceBoxWidget : public vtk3DWidget
{
public:
	static vtkErSliceBoxWidget *New();
	vtkTypeMacro(vtkErSliceBoxWidget, vtk3DWidget);
	void PrintSelf(ostream& os, vtkIndent indent);

	virtual void SetEnabled(int);
	virtual void PlaceWidget(double bounds[6]);
	void UpdatePlacement(void);

	vtkGetMacro(Volume, vtkVolume*);
	void SetVolume(vtkVolume* pVolume);

	void GetPlanes(vtkPlanes* pPlanes);
	
	vtkErSlicePlaneWidget* GetSlicePlaneWidget(int Index);

protected:
	vtkErSliceBoxWidget();
	~vtkErSliceBoxWidget();

	//BTX
	int State;
	enum WidgetState
	{
		Start = 0,
		Moving,
		Scaling,
		Outside
	};
	//ETX

	void CreateDefaultProperties();
    
	// Event handling
	static void ProcessEvents(vtkObject* object, unsigned long event, void* clientdata, void* calldata);

	// Mouse events
	virtual void OnMouseMove();
	virtual void OnLeftButtonDown();
	virtual void OnLeftButtonUp();
	virtual void OnMiddleButtonDown();
	virtual void OnMiddleButtonUp();
	virtual void OnRightButtonDown();
	virtual void OnRightButtonUp();

	virtual void PositionHandles();

	vtkVolume*								Volume;

	// Bounding box lines
	vtkSmartPointer<vtkActor>				BoundingBoxActor;
	vtkSmartPointer<vtkPolyDataMapper>		BoundingBoxMapper;
	vtkSmartPointer<vtkCubeSource>			BoundingBoxSource;
	vtkSmartPointer<vtkProperty>			BoundingBoxProperty;

	// Bounding box points and labels
	vtkSmartPointer<vtkPoints>				BoundingBoxPoints;
	vtkSmartPointer<vtkPolyData>			BoundingBoxPointPolyData;
	vtkSmartPointer<vtkPolyDataMapper>		BoundingBoxPointMapper;
	vtkSmartPointer<vtkActor>				BoundingBoxPointActor;
	vtkSmartPointer<vtkProperty>			BoundingBoxPointProperty;
	vtkSmartPointer<vtkLabeledDataMapper>	BoundingBoxPointLabelMapper;
	vtkSmartPointer<vtkActor2D>				BoundingBoxPointLabelActor;
	vtkSmartPointer<vtkProperty2D>			BoundingBoxPointLabelProperty;

	// Widgets
	vtkSmartPointer<vtkErSlicePlaneWidget>	PointWidget[6];
	vtkSmartPointer<vtkPoints>				DefaultPositions;
	vtkSmartPointer<vtkPoints>				DefaultNormals;

private:
  vtkErSliceBoxWidget(const vtkErSliceBoxWidget&);
  void operator=(const vtkErSliceBoxWidget&);
};


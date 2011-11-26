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

#define VTK_PLANE_OFF 0
#define VTK_PLANE_OUTLINE 1
#define VTK_PLANE_WIREFRAME 2
#define VTK_PLANE_SURFACE 3

#include "vtkErSlicePlane.h"

class VTK_ER_CORE_EXPORT vtkErSlicePlaneWidget : public vtkPolyDataSourceWidget
{
public:
  // Description:
  // Instantiate the object.
  static vtkErSlicePlaneWidget *New();

  vtkTypeMacro(vtkErSlicePlaneWidget, vtkPolyDataSourceWidget);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Methods that satisfy the superclass' API.
  virtual void SetEnabled(int);
  virtual void PlaceWidget(double bounds[6]);
  void PlaceWidget()
    {this->Superclass::PlaceWidget();}
  void PlaceWidget(double xmin, double xmax, double ymin, double ymax, 
                   double zmin, double zmax)
    {this->Superclass::PlaceWidget(xmin,xmax,ymin,ymax,zmin,zmax);}

  // Description:
  // Set/Get the resolution (number of subdivisions) of the plane.
  void SetResolution(int r);
  int GetResolution();

  // Description:
  // Set/Get the origin of the plane.
  void SetOrigin(double x, double y, double z);
  void SetOrigin(double x[3]);
  double* GetOrigin();
  void GetOrigin(double xyz[3]);

  // Description:
  // Set/Get the position of the point defining the first axis of the plane.
  void SetPoint1(double x, double y, double z);
  void SetPoint1(double x[3]);
  double* GetPoint1();
  void GetPoint1(double xyz[3]);
  
  // Description:
  // Set/Get the position of the point defining the second axis of the plane.
  void SetPoint2(double x, double y, double z);
  void SetPoint2(double x[3]);
  double* GetPoint2();
  void GetPoint2(double xyz[3]);

  // Description:
  // Get the center of the plane.
  void SetCenter(double x, double y, double z);
  void SetCenter(double x[3]);
  double* GetCenter();
  void GetCenter(double xyz[3]);

  // Description:
  // Get the normal to the plane.
  void SetNormal(double x, double y, double z);
  void SetNormal(double x[3]);
  double* GetNormal();
  void GetNormal(double xyz[3]);
  
  // Description:
  // Control how the plane appears when GetPolyData() is invoked.
  // If the mode is "outline", then just the outline of the plane
  // is shown. If the mode is "wireframe" then the plane is drawn
  // with the outline plus the interior mesh (corresponding to the
  // resolution specified). If the mode is "surface" then the plane
  // is drawn as a surface.
  vtkSetClampMacro(Representation,int,VTK_PLANE_OFF,VTK_PLANE_SURFACE);
  vtkGetMacro(Representation,int);
  void SetRepresentationToOff()
    {this->SetRepresentation(VTK_PLANE_OFF);}
  void SetRepresentationToOutline()
    {this->SetRepresentation(VTK_PLANE_OUTLINE);}
  void SetRepresentationToWireframe()
    {this->SetRepresentation(VTK_PLANE_WIREFRAME);}
  void SetRepresentationToSurface()
    {this->SetRepresentation(VTK_PLANE_SURFACE);}

  // Description:
  // Force the plane widget to be aligned with one of the x-y-z axes.
  // Remember that when the state changes, a ModifiedEvent is invoked.
  // This can be used to snap the plane to the axes if it is orginally
  // not aligned.
  vtkSetMacro(NormalToXAxis,int);
  vtkGetMacro(NormalToXAxis,int);
  vtkBooleanMacro(NormalToXAxis,int);
  vtkSetMacro(NormalToYAxis,int);
  vtkGetMacro(NormalToYAxis,int);
  vtkBooleanMacro(NormalToYAxis,int);
  vtkSetMacro(NormalToZAxis,int);
  vtkGetMacro(NormalToZAxis,int);
  vtkBooleanMacro(NormalToZAxis,int);

  // Description:
  // Grab the polydata (including points) that defines the plane.  The
  // polydata consists of (res+1)*(res+1) points, and res*res quadrilateral
  // polygons, where res is the resolution of the plane. These point values
  // are guaranteed to be up-to-date when either the InteractionEvent or
  // EndInteraction events are invoked. The user provides the vtkPolyData and
  // the points and polyplane are added to it.
  void GetPolyData(vtkPolyData *pd);

  // Description:
  // Get the planes describing the implicit function defined by the plane
  // widget. The user must provide the instance of the class vtkPlane. Note
  // that vtkPlane is a subclass of vtkImplicitFunction, meaning that it can
  // be used by a variety of filters to perform clipping, cutting, and
  // selection of data.
  void GetPlane(vtkPlane *plane);

  // Description:
  // Satisfies superclass API.  This returns a pointer to the underlying
  // PolyData.  Make changes to this before calling the initial PlaceWidget()
  // to have the initial placement follow suit.  Or, make changes after the
  // widget has been initialised and call UpdatePlacement() to realise.
  vtkPolyDataAlgorithm* GetPolyDataAlgorithm();
   
  // Description:
  // Satisfies superclass API.  This will change the state of the widget to
  // match changes that have been made to the underlying PolyDataSource
  void UpdatePlacement(void);

  // Description:
  // Get the handle properties (the little balls are the handles). The 
  // properties of the handles when selected and normal can be 
  // manipulated.
  vtkGetObjectMacro(HandleProperty,vtkProperty);
  vtkGetObjectMacro(SelectedHandleProperty,vtkProperty);
  
  // Description:
  // Get the plane properties. The properties of the plane when selected 
  // and unselected can be manipulated.
  virtual void SetPlaneProperty(vtkProperty*);
  vtkGetObjectMacro(PlaneProperty,vtkProperty);
  vtkGetObjectMacro(SelectedPlaneProperty,vtkProperty);
  vtkGetObjectMacro(PlaneActor,vtkErSlicePlaneActor);
  
  vtkGetMacro(Volume, vtkVolume*);
//	vtkSetMacro(Volume, vtkVolume*);
	void SetVolume(vtkVolume* pVolume);

protected:
  vtkErSlicePlaneWidget();
  ~vtkErSlicePlaneWidget();

//BTX - manage the state of the widget
  int State;
  enum WidgetState
  {
    Start=0,
    Moving,
    Scaling,
    Pushing,
    Rotating,
    Spinning,
    Outside
  };
//ETX
    
  //handles the events
  static void ProcessEvents(vtkObject* object, 
                            unsigned long event,
                            void* clientdata, 
                            void* calldata);

  // ProcessEvents() dispatches to these methods.
  void OnLeftButtonDown();
  void OnLeftButtonUp();
  void OnMiddleButtonDown();
  void OnMiddleButtonUp();
  void OnRightButtonDown();
  void OnRightButtonUp();
  void OnMouseMove();

  // controlling ivars
  int NormalToXAxis;
  int NormalToYAxis;
  int NormalToZAxis;
  int Representation;
  void SelectRepresentation();

  // the plane
  vtkErSlicePlaneActor          *PlaneActor;
  vtkPolyDataMapper *PlaneMapper;
  vtkPlaneSource    *PlaneSource;
  vtkPolyData       *PlaneOutline;
  void HighlightPlane(int highlight);

  // glyphs representing hot spots (e.g., handles)
  vtkActor          **Handle;
  vtkPolyDataMapper **HandleMapper;
  vtkSphereSource   **HandleGeometry;
  void PositionHandles();
  void HandlesOn(double length);
  void HandlesOff();
  int HighlightHandle(vtkProp *prop); //returns cell id
  virtual void SizeHandles();
  
  // the normal cone
  vtkActor          *ConeActor;
  vtkPolyDataMapper *ConeMapper;
  vtkConeSource     *ConeSource;
  void HighlightNormal(int highlight);

  // the normal line
  vtkActor          *LineActor;
  vtkPolyDataMapper *LineMapper;
  vtkLineSource     *LineSource;

  // the normal cone
  vtkActor          *ConeActor2;
  vtkPolyDataMapper *ConeMapper2;
  vtkConeSource     *ConeSource2;

  // the normal line
  vtkActor          *LineActor2;
  vtkPolyDataMapper *LineMapper2;
  vtkLineSource     *LineSource2;

  // Do the picking
  vtkCellPicker *HandlePicker;
  vtkCellPicker *PlanePicker;
  vtkActor *CurrentHandle;
  
  // Methods to manipulate the hexahedron.
  void MoveOrigin(double *p1, double *p2);
  void MovePoint1(double *p1, double *p2);
  void MovePoint2(double *p1, double *p2);
  void MovePoint3(double *p1, double *p2);
  void Rotate(int X, int Y, double *p1, double *p2, double *vpn);
  void Spin(double *p1, double *p2);
  void Scale(double *p1, double *p2, int X, int Y);
  void Translate(double *p1, double *p2);
  void Push(double *p1, double *p2);
  
  // Plane normal, normalized
  double Normal[3];

  // Transform the hexahedral points (used for rotations)
  vtkTransform *Transform;
  
  // Properties used to control the appearance of selected objects and
  // the manipulator in general.
  vtkProperty *HandleProperty;
  vtkProperty *SelectedHandleProperty;
  vtkProperty *PlaneProperty;
  vtkProperty *SelectedPlaneProperty;
  void CreateDefaultProperties();
  
  void GeneratePlane();

  int    LastPickValid;
  double HandleSizeFactor;
  
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
	
private:
  vtkErSlicePlaneWidget(const vtkErSlicePlaneWidget&);  //Not implemented
  void operator=(const vtkErSlicePlaneWidget&);  //Not implemented
};


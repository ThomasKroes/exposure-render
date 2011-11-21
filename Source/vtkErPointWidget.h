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

#include "Dll.h"

class vtkActor;
class vtkPolyDataMapper;
class vtkCellPicker;
class vtkPolyData;
class vtkProperty;
class vtkErArrowSource;

// http://www.cmake.org/Wiki/VTKWidgets

class EXPOSURE_RENDER_DLL vtkErPointWidget : public vtk3DWidget
{
public:
  // Description:
  // Instantiate this widget
  static vtkErPointWidget *New();

  vtkTypeMacro(vtkErPointWidget,vtk3DWidget);
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
  // Grab the polydata (including points) that defines the point. A
  // single point and a vertex compose the vtkPolyData.
  void GetPolyData(vtkPolyData *pd);
  
  // Description:
  // Set/Get the position of the point. Note that if the position is set
  // outside of the bounding box, it will be clamped to the boundary of
  // the bounding box.
  void SetPosition(double x, double y, double z)
    {this->Cursor3D->SetFocalPoint(x,y,z);}
  void SetPosition(double x[3])
    {this->SetPosition(x[0],x[1],x[2]);}
  double* GetPosition() 
    {return this->Cursor3D->GetFocalPoint();}
  void GetPosition(double xyz[3]) 
    {this->Cursor3D->GetFocalPoint(xyz);}

  // Description:
  // Turn on/off the wireframe bounding box.
  void SetOutline(int o)
    {this->Cursor3D->SetOutline(o);}
  int GetOutline()
    {return this->Cursor3D->GetOutline();}
  void OutlineOn()
    {this->Cursor3D->OutlineOn();}
  void OutlineOff()
    {this->Cursor3D->OutlineOff();}

  // Description:
  // Turn on/off the wireframe x-shadows.
  void SetXShadows(int o)
    {this->Cursor3D->SetXShadows(o);}
  int GetXShadows()
    {return this->Cursor3D->GetXShadows();}
  void XShadowsOn()
    {this->Cursor3D->XShadowsOn();}
  void XShadowsOff()
    {this->Cursor3D->XShadowsOff();}

  // Description:
  // Turn on/off the wireframe y-shadows.
  void SetYShadows(int o)
    {this->Cursor3D->SetYShadows(o);}
  int GetYShadows()
    {return this->Cursor3D->GetYShadows();}
  void YShadowsOn()
    {this->Cursor3D->YShadowsOn();}
  void YShadowsOff()
    {this->Cursor3D->YShadowsOff();}

  // Description:
  // Turn on/off the wireframe z-shadows.
  void SetZShadows(int o)
    {this->Cursor3D->SetZShadows(o);}
  int GetZShadows()
    {return this->Cursor3D->GetZShadows();}
  void ZShadowsOn()
    {this->Cursor3D->ZShadowsOn();}
  void ZShadowsOff()
    {this->Cursor3D->ZShadowsOff();}

  // Description:
  // If translation mode is on, as the widget is moved the bounding box,
  // shadows, and cursor are all translated simultaneously as the point
  // moves.
  void SetTranslationMode(int mode)
    { this->Cursor3D->SetTranslationMode(mode); this->Cursor3D->Update(); }
  int GetTranslationMode()
    { return this->Cursor3D->GetTranslationMode(); }
  void TranslationModeOn()
    { this->SetTranslationMode(1); }
  void TranslationModeOff()
    { this->SetTranslationMode(0); }
  
  // Description:
  // Convenience methods to turn outline and shadows on and off.
  void AllOn()
    {
      this->OutlineOn();
      this->XShadowsOn();
      this->YShadowsOn();
      this->ZShadowsOn();
    }
  void AllOff()
    {
      this->OutlineOff();
      this->XShadowsOff();
      this->YShadowsOff();
      this->ZShadowsOff();
    }

  // Description:
  // Get the handle properties (the little balls are the handles). The 
  // properties of the handles when selected and normal can be 
  // set.
  vtkGetObjectMacro(Property,vtkProperty);
  vtkGetObjectMacro(SelectedProperty,vtkProperty);
  
  // Description:
  // Set the "hot spot" size; i.e., the region around the focus, in which the
  // motion vector is used to control the constrained sliding action. Note the
  // size is specified as a fraction of the length of the diagonal of the 
  // point widget's bounding box.
  vtkSetClampMacro(HotSpotSize,double,0.0,1.0);
  vtkGetMacro(HotSpotSize,double);
  
protected:

  vtkErPointWidget();
  ~vtkErPointWidget();

//BTX - manage the state of the widget
  friend class vtkErAreaLightWidget;
  
  int State;
  enum WidgetState
  {
    Start=0,
    Moving,
    Scaling,
    Translating,
    Outside
  };

    
  // Handles the events
  static void ProcessEvents(vtkObject* object, 
                            unsigned long event,
                            void* clientdata, 
                            void* calldata);

  // ProcessEvents() dispatches to these methods.
  virtual void OnMouseMove();
  virtual void OnLeftButtonDown();
  virtual void OnLeftButtonUp();
  virtual void OnMiddleButtonDown();
  virtual void OnMiddleButtonUp();
  virtual void OnRightButtonDown();
  virtual void OnRightButtonUp();
  
  // the cursor3D
  vtkActor          *Actor;
  vtkPolyDataMapper *Mapper;
  vtkCursor3D       *Cursor3D;
  void Highlight(int highlight);

  // Do the picking
  vtkCellPicker *CursorPicker;
  
  // Methods to manipulate the cursor
  int ConstraintAxis;
  void Translate(double *p1, double *p2);
  void Scale(double *p1, double *p2, int X, int Y);
  void MoveFocus(double *p1, double *p2);
  int TranslationMode;

  // Properties used to control the appearance of selected objects and
  // the manipulator in general.
  vtkProperty *Property;
  vtkProperty *SelectedProperty;
  void CreateDefaultProperties();
  
  // The size of the hot spot.
  double HotSpotSize;
  int DetermineConstraintAxis(int constraint, double *x);
  int WaitingForMotion;
  int WaitCount;
  
	vtkSmartPointer<vtkActor>			XAxisActor;
	vtkSmartPointer<vtkPolyDataMapper>	XAxisMapper;
	vtkSmartPointer<vtkErArrowSource>	XAxisHandleGeometry;
	vtkSmartPointer<vtkProperty>		XAxisProperty;

	vtkSmartPointer<vtkActor>			YAxisActor;
	vtkSmartPointer<vtkPolyDataMapper>	YAxisMapper;
	vtkSmartPointer<vtkErArrowSource>	YAxisHandleGeometry;
	vtkSmartPointer<vtkProperty>		YAxisProperty;

	vtkSmartPointer<vtkActor>			ZAxisActor;
	vtkSmartPointer<vtkPolyDataMapper>	ZAxisMapper;
	vtkSmartPointer<vtkErArrowSource>	ZAxisHandleGeometry;
	vtkSmartPointer<vtkProperty>		ZAxisProperty;
	//ETX
private:
  vtkErPointWidget(const vtkErPointWidget&);  //Not implemented
  void operator=(const vtkErPointWidget&);  //Not implemented
};
/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Stable.h"

#include "RenderOutputDialog.h"

#include <vtkMetaImageReader.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkCylinderSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkAxesActor.h>
#include <vtkCubeSource.h>

QRenderOutputDialog::QRenderOutputDialog(QWidget* pParent) :
	QDialog(pParent),
	m_MainLayout()
{
	setWindowTitle("Render Output");
	setWindowIcon(GetIcon("palette"));

	setLayout(&m_MainLayout);

	m_MainLayout.setContentsMargins(0, 0, 0, 0);

	setWindowFlags(Qt::WindowStaysOnTopHint);

	m_MainLayout.addWidget(&m_QtVtkWidget);

	m_Volume		= vtkVolume::New();
	m_VolumeMapper	= vtkVolumeCudaMapper::New();

	// Create and configure scene renderer
	m_SceneRenderer = vtkRenderer::New();
	m_SceneRenderer->SetBackground(0.25, 0.25, 0.25);
	m_SceneRenderer->SetBackground2(0.25, 0.25, 0.25);
	m_SceneRenderer->SetGradientBackground(true);
	m_SceneRenderer->GetActiveCamera()->SetPosition(0.0, 0.0, 10.0);
	m_SceneRenderer->GetActiveCamera()->SetFocalPoint(0.0, 0.0, 0.0);
//	m_SceneRenderer->GetActiveCamera()->ParallelProjectionOn();

	m_QtVtkWidget.GetRenderWindow()->AddRenderer(m_SceneRenderer);




	// Set up the widget
  vtkSmartPointer<vtkOrientationMarkerWidget> widget =
    vtkSmartPointer<vtkOrientationMarkerWidget>::New();
  widget->SetOutlineColor( 0.9300, 0.5700, 0.1300 );

  vtkCylinderSource *cylinder = vtkCylinderSource::New();
  cylinder->SetResolution(800);

  vtkCubeSource* pBox = vtkCubeSource::New();

  pBox->SetXLength(1.0);
  pBox->SetYLength(1.0);
  pBox->SetZLength(1.0);
  pBox->SetCenter(0.5, 0.5, 0.5);

  // The mapper is responsible for pushing the geometry into the graphics
  // library. It may also do color mapping, if scalars or other attributes
  // are defined.
  vtkPolyDataMapper *cylinderMapper = vtkPolyDataMapper::New();
  cylinderMapper->SetInputConnection(cylinder->GetOutputPort());

  // The actor is a grouping mechanism: besides the geometry (mapper), it
  // also has a property, transformation matrix, and/or texture map.
  // Here we set its color and rotate it -22.5 degrees.
  vtkActor *cylinderActor = vtkActor::New();
  cylinderActor->SetMapper(cylinderMapper);
  cylinderActor->GetProperty()->SetColor(1.0000, 0.3882, 0.2784);
  cylinderActor->GetProperty()->SetOpacity(0.5);
//  cylinderActor->RotateX(30.0);
//  cylinderActor->RotateY(-45.0);
 m_SceneRenderer->AddActor(cylinderActor);


 
 m_SceneRenderer->GetActiveCamera()->SetFocalDisk(0.0f);


	vtkSmartPointer<vtkMetaImageReader> MetaImageReader = vtkMetaImageReader::New();

	MetaImageReader->SetFileName("C://Volumes//engine_small.mhd");

	MetaImageReader->Update();

	m_VolumeMapper->SetInput(MetaImageReader->GetOutput());

    m_Volume->SetMapper(m_VolumeMapper);
//    vtkVolumeProperty* prop = vtkVolumeProperty::New();
//    m_Volume->SetProperty(prop);
    

	m_SceneRenderer->AddViewProp(m_Volume);
};

QRenderOutputDialog::~QRenderOutputDialog(void)
{
}

QSize QRenderOutputDialog::sizeHint() const
{
	return QSize(600, 300);
}

void QRenderOutputDialog::accept()
{
	QDialog::accept();
}
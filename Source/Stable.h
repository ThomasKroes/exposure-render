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

#include <QtGui>
#include <QtXml\qdom.h>
#include <QHttp>

#include "Logger.h"
#include "Status.h"
#include "Controls.h"
#include "Geometry.h"
#include "CudaUtilities.h"

inline void ReadVectorElement(QDomElement& Parent, const QString& Name, float& X, float& Y, float& Z)
{
	X = Parent.firstChildElement(Name).attribute("X").toFloat();
	Y = Parent.firstChildElement(Name).attribute("Y").toFloat();
	Z = Parent.firstChildElement(Name).attribute("Z").toFloat();
}

inline void WriteVectorElement(QDomDocument& DOM, QDomElement& Parent, const QString& Name, const float& X, const float& Y, const float& Z)
{
	QDomElement Vector = DOM.createElement(Name);
	Parent.appendChild(Vector);

	Vector.setAttribute("X", X);
	Vector.setAttribute("Y", Y);
	Vector.setAttribute("Z", Z);
}

inline QIcon GetIcon(const QString& Name)
{
	return QIcon(QApplication::applicationDirPath() + "/Icons/" + Name + ".png");
}

QString GetOpenFileName2(const QString& Caption, const QString& Filter, const QString& Icon);
QString GetSaveFileName(const QString& Caption, const QString& Filter, const QString& Icon);
void SaveImage(const unsigned char* pImageBuffer, const int& Width, const int& Height, QString FilePath = "");


// General VTK stuff
#include <vtkObject.h>
#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>
#include <vtkPiecewiseFunction.h>




// Volume stuff
#include <vtkVolume.h>
#include <vtkVolumeMapper.h>
#include <vtkVolumeProperty.h>
#include <vtkAbstractVolumeMapper.h>

// Renderer stuff
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>

// VTK actors
#include <vtkActor.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>




#include <vtkConeSource.h>
#include <vtkRenderWindow.h>
#include <vtkPolyDataMapper.h>

#include <vtkImageImport.h>
#include <vtkImageActor.h>
#include <vtkInteractorStyleImage.h>
#include <vtkCallbackCommand.h>
#include <vtkCamera.h>

#include <vtkImageData.h>
#include <vtkUnsignedCharArray.h>
#include <vtkPointData.h>

// VTK multi threading
#include <vtkMultiThreader.h>


#include <vtkMetaImageReader.h>
#include <vtkPiecewiseFunction.h>
#include <vtkColorTransferFunction.h>

#include <vtkgl.h>

// VTK image stuff
#include <vtkImageData.h>
#include <vtkImageCast.h>
#include <vtkImageGradientMagnitude.h>


#include <vtkTransform.h>
#include <vtkPerspectiveTransform.h>
#include <vtkCommand.h>
#include <vtkLight.h>
#include <vtkLightCollection.h>
#include <vtkActorCollection.h>
#include <vtkPlaneSource.h>
#include <vtkRenderWindowInteractor.h>
#include <vtk3DWidget.h>
#include <vtkCursor3D.h>
#include <vtkSmartPointer.h>

#include <vtkActor.h>
#include <vtkAssemblyNode.h>
#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkCellPicker.h>
#include <vtkMath.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderWindow.h>
#include <vtkObjectFactory.h>

#include <vtkLight.h>

#include <vtkOpenGLCamera.h>

#include <vtkObjectFactory.h>
#include <vtkRenderer.h>
#include <vtkBoundingBox.h>
#include <vtkMath.h>

#include <vtkPolyDataAlgorithm.h>

#include <vtkAppendPolyData.h>
#include <vtkConeSource.h>
#include <vtkCylinderSource.h>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkObjectFactory.h>
#include <vtkPolyData.h>
#include <vtkTransform.h>
#include <vtkTransformFilter.h>

#include <vtk3DWidget.h>
#include <vtkSmartPointer.h>
#include <vtkLineSource.h>

#include <vtkActor.h>
#include <vtkAssemblyNode.h>
#include <vtkAssemblyPath.h>
#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkCellPicker.h>
#include <vtkCommand.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkMath.h>
#include <vtkObjectFactory.h>
#include <vtkPlanes.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSphereSource.h>

#include <vtkLightActor.h>
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

#include <QVTKWidget.h>

// VTK
#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkConeSource.h>
#include <vtkRenderWindow.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkImageImport.h>
#include <vtkImageActor.h>
#include <vtkInteractorStyleImage.h>
#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkVolumeMapper.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkImageData.h>
#include <vtkUnsignedCharArray.h>
#include <vtkPointData.h>
#include <vtkPiecewiseFunction.h>
#include <vtkVolume.h>

#include "VtkErVolumeMapper.h"
#include "VtkErVolumeProperty.h"
#include "VtkErCamera.h"
#include "VtkErVolume.h"
#include "vtkErRenderCanvas.h"

#include <vtkVolume.h>

// http://www.na-mic.org/svn/Slicer3/branches/cuda/Modules/VolumeRenderingCuda/

class CVtkRenderWidget : public QDialog
{
    Q_OBJECT

public:
    CVtkRenderWidget(QWidget* pParent = NULL);
	
	QVTKWidget*		GetQtVtkWidget(void);

public slots:
	void OnRenderBegin(void);
	void OnRenderEnd(void);
	void SetActive(void);
	void LoadVolume(const QString& FilePath);
	void OnViewFront(void);
	void OnViewBack(void);
	void OnViewLeft(void);
	void OnViewRight(void);
	void OnViewTop(void);
	void OnViewBottom(void);

	vtkVolume* GetVolume(void) { return m_Volume.GetPointer(); };
	vtkErVolumeProperty* GetVolumeProperty(void) { return m_VolumeProperty.GetPointer(); };
	vtkErVolumeMapper* GetVolumeMapper(void) { return m_VolumeMapper.GetPointer(); };
	vtkRenderer* GetRenderer(void) { return m_Renderer.GetPointer(); };
	vtkErCamera* GetCamera(void) { return m_Camera.GetPointer(); }

private:
	QGridLayout								m_MainLayout;
	QPushButton								m_ViewFront;
	QPushButton								m_ViewBack;
	QPushButton								m_ViewLeft;
	QPushButton								m_ViewRight;
	QPushButton								m_ViewTop;
	QPushButton								m_ViewBottom;
	QVTKWidget								m_QtVtkWidget;

	void SetupVtk(void);

public:
	vtkSmartPointer<vtkVolume>				m_Volume;
	vtkSmartPointer<vtkErVolumeProperty>	m_VolumeProperty;
	vtkSmartPointer<vtkErVolumeMapper>		m_VolumeMapper;
	vtkSmartPointer<vtkRenderer>			m_Renderer;
	vtkSmartPointer<vtkRenderer>			m_OverlayRenderer;
	vtkSmartPointer<vtkCallbackCommand>		m_TimerCallback;
	vtkSmartPointer<vtkErCamera>			m_Camera;
	vtkSmartPointer<vtkErRenderCanvas>		m_RenderCanvas;
};

extern CVtkRenderWidget* gpActiveRenderWidget;

class QRenderView : public QGraphicsView
{
    Q_OBJECT

public:
    QRenderView(QWidget* pParent = NULL);

	QGraphicsScene		m_GraphicsScene;
	CVtkRenderWidget	m_RenderWidget;
};

#include <QVTKGraphicsItem.h>
#include <QGraphicsView>
#include <QResizeEvent>
#include "QVTKWidget2.h"
#include "vtkGenericOpenGLRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkTextActor3D.h"
#include <vtkGraphLayoutView.h>

class QTestGraphicsItem : public QVTKGraphicsItem
{
public:
  QTestGraphicsItem(QGLContext* ctx, QGraphicsItem* p=0)
  : QVTKGraphicsItem(ctx, p)
{//
	GraphLayoutView = vtkGraphLayoutView::New();
//  GraphLayoutView.TakeReference(vtkGraphLayoutView::New());
  
  
GraphLayoutView->SetInteractor(this->GetInteractor());
GraphLayoutView->SetRenderWindow(this->GetRenderWindow());
  GraphLayoutView->ResetCamera();

}

  ~QTestGraphicsItem(){};

protected:
  vtkSmartPointer<vtkGraphLayoutView> GraphLayoutView;
};

class OpenGLScene : public QGraphicsScene
{
  Q_OBJECT
  public:
	  OpenGLScene(QGLContext* ctx, QObject* p=0)  : QGraphicsScene(p), mContext(ctx)
	  {
		  this->addItem(new QTestGraphicsItem(ctx, NULL));
	  };
	  ~OpenGLScene(){};

  Q_SIGNALS:
    void enterState1();
    void enterState2();
    void enterState3();
    void enterState4();

  protected:
    QGLContext* mContext;
    QStateMachine machine;
    QGraphicsWidget* mGraphLayoutView;
    QGraphicsWidget* mTreeRingView;
    QGraphicsWidget* mWebView;
    int CurrentState;

    void mousePressEvent(QGraphicsSceneMouseEvent* e){};

};

class GraphicsView : public QGraphicsView
{
  public:
    GraphicsView()
    {
      mCtx = new QGLContext(QGLFormat());
      mWidget = new QVTKWidget2(mCtx);
      this->setViewport(mWidget);
//      this->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
      this->setScene(new OpenGLScene(mCtx, this));
      vtkSmartPointer<vtkRenderer> ren = vtkSmartPointer<vtkRenderer>::New();
      ren->SetBackground(0,0,0);
      ren->SetBackground2(1,1,1);
      ren->SetGradientBackground(1);
      vtkSmartPointer<vtkTextActor3D> textActor = vtkSmartPointer<vtkTextActor3D>::New();
      textActor->SetInput("Qt & VTK!!");
      ren->AddViewProp(textActor);
      ren->ResetCamera();
      mWidget->GetRenderWindow()->AddRenderer(ren);
      mWidget->GetRenderWindow()->SetSwapBuffers(0);  // don't let VTK swap buffers on us
      mWidget->setAutoBufferSwap(true);
    }
    ~GraphicsView()
    {
    }

  protected:

    void drawBackground(QPainter* p, const QRectF& vtkNotUsed(r))
      {
#if QT_VERSION >= 0x040600
      p->beginNativePainting();
#endif
      mWidget->GetRenderWindow()->PushState();
      mWidget->GetRenderWindow()->Render();
      mWidget->GetRenderWindow()->PopState();
#if QT_VERSION >= 0x040600
      p->endNativePainting();
#endif
      }

    void resizeEvent(QResizeEvent *event)
      {
        // give the same size to the scene that his widget has
        if (scene())
            scene()->setSceneRect(QRect(QPoint(0, 0), event->size()));
        QGraphicsView::resizeEvent(event);
        mWidget->GetRenderWindow()->SetSize(event->size().width(), event->size().height());
      }
    QGLContext* mCtx;
    QVTKWidget2* mWidget;
};
#pragma once

#include <QtGui>
#include <QVTKWidget.h>

// VTK
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

// Interactor
#include "InteractorStyleRealisticCamera.h"

class CVtkWidget : public QWidget
{
    Q_OBJECT

public:
    CVtkWidget(QWidget* pParent = NULL);
	
	QVTKWidget*		GetQtVtkWidget(void);

public slots:
	void OnRenderBegin(void);
	void OnRenderEnd(void);
	void OnPreRenderFrame(void);
	void OnPostRenderFrame(void);
	void OnResize(void);
	void OnRenderLoopTimer(void);

private:
	void SetupRenderView(void);

	QGridLayout							m_MainLayout;
	QVTKWidget							m_QtVtkWidget;

	vtkImageImport*						m_pImageImport;
	vtkImageActor*						m_pImageActor;
	vtkInteractorStyleImage*			m_pInteractorStyleImage;
	vtkRenderer*						m_pSceneRenderer;
	vtkRenderWindow*					m_pRenderWindow;
	vtkRenderWindowInteractor*			m_pRenderWindowInteractor;
	vtkCallbackCommand*					m_pKeyPressCallback;
	vtkCallbackCommand*					m_pKeyReleaseCallback;
	CInteractorStyleRealisticCamera*	m_pInteractorStyleRealisticCamera;

	QTimer								m_RenderLoopTimer;
};
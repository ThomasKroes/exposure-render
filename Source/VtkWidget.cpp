
#include "VtkWidget.h"
#include "RenderThread.h"

// Key press callback
void KeyPressCallbackFunction(vtkObject* pCaller, long unsigned int EventId, void* pClientData, void* pCallData)
{
	/*
	vtkRenderWindowInteractor* pRenderWindowInteractor = static_cast<vtkRenderWindowInteractor*>(pCaller);

	char* pKeySymbol = pRenderWindowInteractor->GetKeySym();

	if (strcmp(pKeySymbol, "space") == 0)
	{
		pRenderWindowInteractor->SetInteractorStyle(gpMainWindow->m_pInteractorStyleImage);

		// Change the cursor to a pointing, thus indicating the change in interaction mode
		gpMainWindow->setCursor(QCursor(Qt::CursorShape::PointingHandCursor));
	}
	*/
}

// Key press callback
void KeyReleaseCallbackFunction(vtkObject* pCaller, long unsigned int EventId, void* pClientData, void* pCallData)
{
	/*
	vtkRenderWindowInteractor* pRenderWindowInteractor = static_cast<vtkRenderWindowInteractor*>(pCaller);

	char* pKeySymbol = pRenderWindowInteractor->GetKeySym();

	if (strcmp(pKeySymbol, "space") == 0)
	{
		pRenderWindowInteractor->SetInteractorStyle(gpMainWindow->m_pInteractorStyleRealisticCamera);

		// Change the cursor to a pointing, thus indicating the change in interaction mode
		gpMainWindow->setCursor(QCursor(Qt::CursorShape::ArrowCursor));
	}
	*/
}

CVtkWidget::CVtkWidget(QWidget* pParent) :
	QWidget(pParent),
	m_MainLayout(),
	m_QtVtkWidget(),
	m_RenderLoopTimer()
{
	// Create and apply main layout
	setLayout(&m_MainLayout);

	// Add VTK widget 
	m_MainLayout.addWidget(&m_QtVtkWidget);

	// Notify us when rendering begins and ends, before/after each rendered frame, when stuff becomes dirty, and when timer has timed out
	connect(&gRenderStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
	connect(&gRenderStatus, SIGNAL(RenderEnd()), this, SLOT(OnRenderEnd()));
	connect(&gRenderStatus, SIGNAL(PreRenderFrame()), this, SLOT(OnPreRenderFrame()));
	connect(&gRenderStatus, SIGNAL(PostRenderFrame()), this, SLOT(OnPostRenderFrame()));
	connect(&gRenderStatus, SIGNAL(Dirty(int)), this, SLOT(OnDirty(int)));
	connect(&m_RenderLoopTimer, SIGNAL(timeout()), this, SLOT(OnRenderLoopTimer()));

	// Setup the render view
	SetupRenderView();
}

QVTKWidget* CVtkWidget::GetQtVtkWidget(void)
{
	return &m_QtVtkWidget;
}

void CVtkWidget::OnRenderBegin(void)
{
	if (!Scene())
		return;

	// Create and configure image importer
	m_pImageImport = vtkImageImport::New();
	m_pImageImport->SetDataSpacing(1, 1, 1);
	m_pImageImport->SetDataOrigin(-400, -300, 0);
	m_pImageImport->SetImportVoidPointer((void*)malloc(3 * 800 * 600 * sizeof(unsigned char)));
	m_pImageImport->SetWholeExtent(0, 800 - 1, 0, 600 - 1, 0, 0);
	m_pImageImport->SetDataExtentToWholeExtent();
	m_pImageImport->SetDataScalarTypeToUnsignedChar();
	m_pImageImport->SetNumberOfScalarComponents(3);
	m_pImageImport->Update();

	// Create and configure background image actor
	m_pImageActor = vtkImageActor::New();
	m_pImageActor->SetInterpolate(1);
	m_pImageActor->SetInput(m_pImageImport->GetOutput());
	m_pImageActor->SetScale(1, -1, -1);
	m_pImageActor->VisibilityOn();

	// Add the image actor
	m_pSceneRenderer->AddActor(m_pImageActor);

	// Scale
	m_pSceneRenderer->GetActiveCamera()->SetParallelScale(600.0f);
	/**/

	// Start the timer
	m_RenderLoopTimer.start(30);
}

void CVtkWidget::OnRenderEnd(void)
{
	if (!Scene())
		return;

	m_pSceneRenderer->RemoveActor(m_pImageActor);

	m_pImageImport = NULL;
	m_pImageActor = NULL;

	// Stop the timer
	m_RenderLoopTimer.start(30);
}

void CVtkWidget::OnPreRenderFrame(void)
{
	if (!Scene())
		return;

}

void CVtkWidget::OnPostRenderFrame(void)
{
	if (!Scene())
		return;

	
	
	
}

void CVtkWidget::SetupRenderView(void)
{
	// Create and configure scene renderer
	m_pSceneRenderer = vtkRenderer::New();
	m_pSceneRenderer->SetBackground(0.4, 0.4, 0.43);
	m_pSceneRenderer->SetBackground2(0.9, 0.9, 0.9);
	m_pSceneRenderer->SetGradientBackground(true);
	m_pSceneRenderer->GetActiveCamera()->SetPosition(0.0, 0.0, 1.0);
	m_pSceneRenderer->GetActiveCamera()->SetFocalPoint(0.0, 0.0, 0.0);
	m_pSceneRenderer->GetActiveCamera()->ParallelProjectionOn();

	// Get render window and configure
	m_pRenderWindow = GetQtVtkWidget()->GetRenderWindow();
	m_pRenderWindow->AddRenderer(m_pSceneRenderer);

	// Key press callback
	m_pKeyPressCallback = vtkCallbackCommand::New();
	m_pKeyPressCallback->SetCallback(KeyPressCallbackFunction);

	// Key press callback
	m_pKeyReleaseCallback = vtkCallbackCommand::New();
	m_pKeyReleaseCallback->SetCallback(KeyReleaseCallbackFunction);

	// Create interactor style for camera navigation
	m_pInteractorStyleRealisticCamera = CInteractorStyleRealisticCamera::New();
	m_pInteractorStyleImage = vtkInteractorStyleImage::New();

	// Add observers
	m_pRenderWindow->GetInteractor()->SetInteractorStyle(m_pInteractorStyleRealisticCamera);
	//	m_pRenderWindow->GetInteractor()->SetInteractorStyle(m_pInteractorStyleImage);

	m_pRenderWindow->GetInteractor()->AddObserver(vtkCommand::KeyPressEvent, m_pKeyPressCallback);
	m_pRenderWindow->GetInteractor()->AddObserver(vtkCommand::KeyReleaseEvent, m_pKeyReleaseCallback);
}

void CVtkWidget::OnDirty(int Dirty)
{
	if (!Scene())
		return;


}

void CVtkWidget::OnRenderLoopTimer(void)
{
	m_pImageImport->SetDataExtent(0, Scene()->m_Camera.m_Film.m_Resolution.Width() - 1, 0, Scene()->m_Camera.m_Film.m_Resolution.Height() - 1, 0, 0);
	m_pImageImport->SetWholeExtent(0, Scene()->m_Camera.m_Film.m_Resolution.Width() - 1, 0, Scene()->m_Camera.m_Film.m_Resolution.Height() - 1, 0, 0);
	m_pImageActor->SetDisplayExtent(0, Scene()->m_Camera.m_Film.m_Resolution.Width() - 1, 0, Scene()->m_Camera.m_Film.m_Resolution.Height() - 1, 0, 0);

	//	m_pImageImport->setup(0, gpScene->m_Camera.m_Film.m_Resolution.Width() - 1, 0, gpScene->m_Camera.m_Film.m_Resolution.Height() - 1, 0, 0);
	m_pImageImport->Update();
	m_pImageImport->SetImportVoidPointer(NULL);
	m_pImageImport->SetImportVoidPointer(gpRenderThread->GetRenderImage());
/*
	m_pImageActor->SetInput(m_pImageImport->GetOutput());
	// 	m_pImageActor->VisibilityOn();
*/
	m_pRenderWindow->GetInteractor()->Render();
}

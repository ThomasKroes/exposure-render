
#include "RenderThread.h"
#include "LoadVolume.h"
#include "Scene.h"
#include "MainWindow.h"

// Qt
#include <QtGui>

// CUDA
#include "VolumeTracer.cuh"

// VTK
#include <vtkImageData.h>

CRenderThread* gpRenderThread = NULL;

bool InitializeCuda(void)
{
	if (cudaSetDevice(cutGetMaxGflopsDeviceId()) != cudaSuccess)
		return false;

	return true;
}

class CCudaTimer
{
public:
	CCudaTimer(void)
	{
		StartTimer();
	}

	virtual ~CCudaTimer(void)
	{
		if (m_Started)
			StopTimer();
	}

	void StartTimer(void)
	{
		cudaEventCreate(&m_EventStart);
		cudaEventCreate(&m_EventStop);
		cudaEventRecord(m_EventStart, 0);

		m_Started = true;
	}

	float StopTimer(void)
	{
		if (!m_Started)
			return 0.0f;

		cudaEventRecord(m_EventStop, 0);
		cudaEventSynchronize(m_EventStop);

		float TimeDelta = 0.0f;

		cudaEventElapsedTime(&TimeDelta, m_EventStart, m_EventStop);
		cudaEventDestroy(m_EventStart);
		cudaEventDestroy(m_EventStop);

		m_Started = false;

		return TimeDelta;
	}

private:
	bool			m_Started;
	cudaEvent_t 	m_EventStart;
	cudaEvent_t 	m_EventStop;
};

CRenderThread::CRenderThread(const QString& FileName, QObject* pParent) :
	QThread(pParent),
	m_FileName(FileName),
	m_N(0),
	m_pRenderImage(NULL),
	m_pImageDataVolume(NULL),
	m_pDevScene(NULL),
	m_pDevRandomStates(NULL),
	m_pDevAccEstXyz(NULL),
	m_pDevEstFrameXyz(NULL),
	m_pDevEstFrameBlurXyz(NULL),
	m_pDevEstRgbLdr(NULL),
	m_pImageCanvas(NULL),
	m_SizeVolume(0),
	m_SizeHdrAccumulationBuffer(0),
	m_SizeHdrFrameBuffer(0),
	m_SizeHdrBlurFrameBuffer(0),
	m_SizeLdrFrameBuffer(0),
	m_Abort(false)
{
}

CRenderThread::~CRenderThread(void)
{
}

void CRenderThread::run()
{
	if (!InitializeCuda())
	{
		// Create message box, indicating that CUDA cannot be initialized
		QMessageBox MessageBox(QMessageBox::Icon::Critical, "CUDA error", "Could not initialize CUDA, this application will now exit");

		// Make it a modal message box
		MessageBox.setWindowModality(Qt::WindowModal);

		// Show it
		MessageBox.exec();
		
		return;
	}

	// Inform others when the rendering begins and ends
	connect(this, SIGNAL(RenderEnd()), gpMainWindow, SLOT(OnRenderEnd()));
	connect(this, SIGNAL(RenderBegin()), gpMainWindow, SLOT(OnRenderBegin()));

	m_Scene.m_Camera.m_Film.m_Resolution.Set(Vec2i(800, 600));
	m_Scene.m_Camera.m_Aperture.m_Size = 0.01f;
	m_Scene.m_Camera.m_Focus.m_FocalDistance = (m_Scene.m_Camera.m_Target - m_Scene.m_Camera.m_From).Length();
	m_Scene.m_Camera.m_SceneBoundingBox = m_Scene.m_BoundingBox;
	m_Scene.m_Camera.SetViewMode(ViewModeFront);
	m_Scene.m_Camera.Update();

	// Bind the volume
	BindVolumeData((short*)m_pImageDataVolume->GetScalarPointer(), m_Scene.m_Resolution);

	// Allocate CUDA memory for scene
	cudaMalloc((void**)&m_pDevScene, sizeof(CScene));

	// Let others know that we are starting with rendering
	emit RenderBegin();

	while (!m_Abort)
	{
		// Let others know we are starting with a new frame
		emit PreFrame();

		// CUDA time for profiling
		CCudaTimer CudaTimer;

		// Make a local copy of the scene, this to prevent modification to the scene from outside this thread
		CScene SceneCopy = m_Scene;

		// Update the camera, do not remove
		SceneCopy.m_Camera.Update();

		// Copy scene from host to device
		cudaMemcpy(m_pDevScene, &SceneCopy, sizeof(CScene), cudaMemcpyHostToDevice);

		// Resizing the image canvas requires special attention
		if (SceneCopy.m_DirtyFlags.HasFlag(FilmResolutionDirty))
		{
			// Allocate host image buffer, this thread will blit it's frames to this buffer
			m_pRenderImage = (unsigned char*)malloc(3 * m_Scene.m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(unsigned char));
			memset(m_pRenderImage, 0, 3 * m_Scene.m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(unsigned char));
			
			// Compute size of the CUDA buffers
			m_SizeVolume				= m_Scene.m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(curandStateXORWOW_t);
			m_SizeHdrAccumulationBuffer	= m_Scene.m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(CColorXyz);
			m_SizeHdrFrameBuffer		= m_Scene.m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(CColorXyz);
			m_SizeHdrBlurFrameBuffer	= m_Scene.m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(CColorXyz);
			m_SizeLdrFrameBuffer		= 3 * m_Scene.m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(unsigned char);

			// Allocate device buffers
			cudaMalloc((void**)&m_pDevRandomStates, m_SizeVolume);
			cudaMalloc((void**)&m_pDevAccEstXyz, m_SizeHdrAccumulationBuffer);
			cudaMalloc((void**)&m_pDevEstFrameXyz, m_SizeHdrFrameBuffer);
			cudaMalloc((void**)&m_pDevEstFrameBlurXyz, m_SizeHdrBlurFrameBuffer);
			cudaMalloc((void**)&m_pDevEstRgbLdr, m_SizeLdrFrameBuffer);
			
			// Setup the CUDA random number generator
			SetupRNG(&SceneCopy, m_pDevScene, m_pDevRandomStates);
			
			// Reset buffers to black
			cudaMemset(m_pDevAccEstXyz, 0, SceneCopy.m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(CColorXyz));
			cudaMemset(m_pDevEstFrameXyz, 0, SceneCopy.m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(CColorXyz));
			cudaMemset(m_pDevEstFrameBlurXyz, 0, SceneCopy.m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(CColorXyz));
			cudaMemset(m_pDevEstRgbLdr, 0, SceneCopy.m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(CColorRgbLdr));

			// Reset no. iterations
			m_N = 0.0f;

			// Inform others about the memory allocations
			emit MemoryAllocate();
		}

		// Restart the rendering when when the camera, lights and render params are dirty
		if (SceneCopy.m_DirtyFlags.HasFlag(CameraDirty | LightsDirty | RenderParamsDirty | TransferFunctionDirty))
		{
			// Reset buffers to black
			cudaMemset(m_pDevAccEstXyz, 0, SceneCopy.m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(CColorXyz));
			cudaMemset(m_pDevEstFrameXyz, 0, SceneCopy.m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(CColorXyz));
			cudaMemset(m_pDevEstFrameBlurXyz, 0, SceneCopy.m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(CColorXyz));
			cudaMemset(m_pDevEstRgbLdr, 0, SceneCopy.m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(CColorRgbLdr));

			// Reset no. iterations
			m_N = 0.0f;
		}

		// At this point, all dirty flags should have been taken care of, since the flags in the original scene are now cleared
		m_Scene.m_DirtyFlags.ClearAllFlags();

		// Execute the rendering kernels
		RenderVolume(&SceneCopy, m_pDevScene, m_pDevRandomStates, m_pDevEstFrameXyz);
		BlurImageXyz(m_pDevEstFrameXyz, m_pDevEstFrameBlurXyz, CResolution2D(SceneCopy.m_Camera.m_Film.m_Resolution.Width(), SceneCopy.m_Camera.m_Film.m_Resolution.Height()), 1.3f);
		ComputeEstimate(SceneCopy.m_Camera.m_Film.m_Resolution.Width(), SceneCopy.m_Camera.m_Film.m_Resolution.Height(), m_pDevEstFrameXyz, m_pDevAccEstXyz, m_N, 100.0f, m_pDevEstRgbLdr);

		// Increase the number of iterations performed so far
		m_N++;

		// Blit
		cudaMemcpy(m_pRenderImage, m_pDevEstRgbLdr, 3 * SceneCopy.m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(unsigned char), cudaMemcpyDeviceToHost);

		m_Scene.m_FPS.AddDuration(1000.0f / CudaTimer.StopTimer());

		// Let others know we are finished with a frame
		emit PostFrame();
	}

	// Let others know that we have stopped rendering
	emit RenderEnd();

	// Inform others when the rendering begins and ends
	connect(this, SIGNAL(RenderEnd()), gpMainWindow, SLOT(OnRenderEnd()));
	connect(this, SIGNAL(RenderBegin()), gpMainWindow, SLOT(OnRenderBegin()));

	// Free CUDA buffers
	cudaFree(m_pDevScene);
	cudaFree(m_pDevRandomStates);
	cudaFree(m_pDevAccEstXyz);
	cudaFree(m_pDevEstFrameXyz);
	cudaFree(m_pDevEstFrameBlurXyz);
	cudaFree(m_pDevEstRgbLdr);

	m_pDevRandomStates		= NULL;
	m_pDevAccEstXyz			= NULL;
	m_pDevEstFrameXyz		= NULL;
	m_pDevEstFrameBlurXyz	= NULL;
	m_pDevEstRgbLdr			= NULL;

	// Free render image buffer
	free(m_pRenderImage);
}

void CRenderThread::OnCloseRenderThread(void)
{
	qDebug("Closing render thread");
	m_Abort = true;
}

CScene* Scene(void)
{
	if (gpRenderThread)
		return &gpRenderThread->m_Scene;

	return NULL;
}

void StartRenderThread(const QString& FileName)
{
	// Create new render thread
	gpRenderThread = new CRenderThread(FileName, NULL);

	// Load the VTK volume
	if (!LoadVtkVolume(FileName.toAscii().data(), &gpRenderThread->m_Scene, gpRenderThread->m_pImageDataVolume))
	{
		qDebug("Unable to load VTK volume");
		return;
	}
	else
	{
		qDebug("VTK volume loaded successfully");
	}

	// Force the render thread to allocate the necessary buffers, do not remove this line
	gpRenderThread->m_Scene.m_DirtyFlags.SetFlag(FilmResolutionDirty | CameraDirty);

	// Start the render thread
	gpRenderThread->start();

	qDebug("Render thread started");
}

void KillRenderThread(void)
{
	if (!gpRenderThread)
		return;

	// Kill the render thread
	gpRenderThread->OnCloseRenderThread();

	// Wait for thread to end
	gpRenderThread->wait();

	// Remove the render thread
	delete gpRenderThread;
	gpRenderThread = NULL;

	qDebug("Render thread killed");
}

#include "RenderThread.h"
#include "LoadVolume.h"

// CUDA
#include "VolumeTracer.cuh"

// Qt
#include <QtGui>

// VTK
#include <vtkImageData.h>

// Scene singleton
CScene* gpScene = NULL;

bool gThreadAlive = false;

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
	m_Loaded(false),
	m_Mutex(),
    m_Condition(),
	m_Statistics(),
	m_N(0),
	m_pRenderImage(NULL),
	m_pImageDataVolume(NULL),
	m_pDevScene(NULL),
	m_pDevRandomStates(NULL),
	m_pDevAccEstXyz(NULL),
	m_pDevEstFrameXyz(NULL),
	m_pDevEstFrameBlurXyz(NULL),
	m_pDevEstRgbLdr(NULL),
	m_pImageCanvas(NULL)
{
}

CRenderThread::~CRenderThread(void)
{
    m_Mutex.lock();
    m_Condition.wakeOne();
    m_Mutex.unlock();

    wait();
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

	gpScene->m_Camera.m_Aperture.m_Size = 0.01f;
	gpScene->m_Camera.m_Focus.m_FocalDistance = (gpScene->m_Camera.m_Target - gpScene->m_Camera.m_From).Length();
	gpScene->m_Camera.m_SceneBoundingBox = gpScene->m_BoundingBox;
	gpScene->m_Camera.SetViewMode(ViewModeFront);
	gpScene->m_Camera.Update();

	// Bind the volume
	BindVolumeData((short*)m_pImageDataVolume->GetScalarPointer(), gpScene->m_Resolution);

	// Allocate CUDA memory for scene
	cudaMalloc((void**)&m_pDevScene, sizeof(CScene));

	// Let others know that we are starting with rendering
	emit RenderBegin();

	while (gThreadAlive)
	{
		// CUDA time for profiling
		CCudaTimer CudaTimer;

		// Make a local copy of the scene, this to prevent modification to the scene from outside this thread
		CScene Scene = *gpScene;

		// Update the camera, do not remove
		Scene.m_Camera.Update();

		// Copy scene from host to device
		cudaMemcpy(m_pDevScene, &Scene, sizeof(CScene), cudaMemcpyHostToDevice);

		// Resizing the image canvas requires special attention
		if (Scene.m_DirtyFlags.HasFlag(FilmResolutionDirty))
		{
			// Allocate host image buffer, this thread will blit it's frames to this buffer
			m_pRenderImage = (unsigned char*)malloc(3 * gpScene->m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(unsigned char));
			memset(m_pRenderImage, 0, 3 * gpScene->m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(unsigned char));
			
			// Compute size of the CUDA buffers
			const int SizeRandomStates		= gpScene->m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(curandStateXORWOW_t);
			const int SizeAccEstXyz			= gpScene->m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(CColorXyz);
			const int SizeEstFrameXyz		= gpScene->m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(CColorXyz);
			const int SizeEstFrameBlurXyz	= gpScene->m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(CColorXyz);
			const int SizeEstRgbLdr			= 3 * gpScene->m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(unsigned char);

			// Allocate device buffers
			cudaMalloc((void**)&m_pDevRandomStates, SizeRandomStates);
			cudaMalloc((void**)&m_pDevAccEstXyz, SizeAccEstXyz);
			cudaMalloc((void**)&m_pDevEstFrameXyz, SizeEstFrameXyz);
			cudaMalloc((void**)&m_pDevEstFrameBlurXyz, SizeEstFrameBlurXyz);
			cudaMalloc((void**)&m_pDevEstRgbLdr, SizeEstRgbLdr);
			
			// Setup the CUDA random number generator
			SetupRNG(&Scene, m_pDevScene, m_pDevRandomStates);
			
			// Reset buffers to black
			cudaMemset(m_pDevAccEstXyz, 0, Scene.m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(CColorXyz));
			cudaMemset(m_pDevEstFrameXyz, 0, Scene.m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(CColorXyz));
			cudaMemset(m_pDevEstFrameBlurXyz, 0, Scene.m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(CColorXyz));
			cudaMemset(m_pDevEstRgbLdr, 0, Scene.m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(CColorXyz));

			// Reset no. iterations
			m_N = 0.0f;
		}

		// Restart the rendering when when the camera, lights and render params are dirty
		if (Scene.m_DirtyFlags.HasFlag(CameraDirty | LightsDirty | RenderParamsDirty | TransferFunctionDirty))
		{
			// Reset buffers to black
			cudaMemset(m_pDevAccEstXyz, 0, Scene.m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(CColorXyz));
			cudaMemset(m_pDevEstFrameXyz, 0, Scene.m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(CColorXyz));
			cudaMemset(m_pDevEstFrameBlurXyz, 0, Scene.m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(CColorXyz));
			cudaMemset(m_pDevEstRgbLdr, 0, Scene.m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(CColorXyz));

			// Reset no. iterations
			m_N = 0.0f;
		}

		// At this point, all dirty flags should have been taken care of, since the flags in the original scene are now cleared
		gpScene->m_DirtyFlags.ClearAllFlags();

		// Execute the rendering kernels
		RenderVolume(&Scene, m_pDevScene, m_pDevRandomStates, m_pDevEstFrameXyz);
		BlurImageXyz(m_pDevEstFrameXyz, m_pDevEstFrameBlurXyz, CResolution2D(Scene.m_Camera.m_Film.m_Resolution.Width(), Scene.m_Camera.m_Film.m_Resolution.Height()), 1.3f);
		ComputeEstimate(Scene.m_Camera.m_Film.m_Resolution.Width(), Scene.m_Camera.m_Film.m_Resolution.Height(), m_pDevEstFrameXyz, m_pDevAccEstXyz, m_N, 100.0f, m_pDevEstRgbLdr);

		// Increase the number of iterations performed so far
		m_N++;

		// Blit
		cudaMemcpy(m_pRenderImage, m_pDevEstRgbLdr, 3 * Scene.m_Camera.m_Film.m_Resolution.m_NoElements * sizeof(unsigned char), cudaMemcpyDeviceToHost);

		gpScene->m_FPS.AddDuration(1000.0f / CudaTimer.StopTimer());

		// Inform other about our performance
		emit UpdateFPS(gpScene->m_FPS.m_FilteredDuration);
	}

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

	// Delete the scene
	delete gpScene;
	gpScene = NULL;

	// Free render image buffer
	free(m_pRenderImage);

	// Let others know that we have stopped rendering
	emit RenderEnd();
}
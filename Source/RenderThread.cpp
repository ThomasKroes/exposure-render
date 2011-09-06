
#include "RenderThread.h"
#include "Scene.h"
#include "MainWindow.h"

// CUDA
#include "VolumeTracer.cuh"

// VTK
#include <vtkSmartPointer.h>
#include <vtkMetaImageReader.h>
#include <vtkImageCast.h>
#include <vtkImageResample.h>
#include <vtkImageData.h>
#include <vtkImageGradientMagnitude.h>
#include <vtkCallbackCommand.h>
#include <vtkImageAccumulate.h>
#include <vtkIntArray.h>

// Render status singleton
CRenderStatus gRenderStatus;

// Render thread
CRenderThread* gpRenderThread = NULL;

bool InitializeCuda(void)
{
	/*
	// No CUDA enabled devices
	int NoDevices = 0;

	cudaError_t ErrorID = cudaGetDeviceCount(&NoDevices);

	if (ErrorID != cudaSuccess)
		throw QString("Unable to initialize CUDA");

	// This function call returns 0 if there are no CUDA capable devices.
	if (NoDevices == 0)
		throw QString("There is no device supporting CUDA");
	*/

	return cudaSetDevice(cutGetMaxGflopsDeviceId()) == cudaSuccess;
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

QProgressDialog* gpProgressDialog = NULL;

void OnProgress(vtkObject* pCaller, long unsigned int EventId, void* pClientData, void* CallData)
{
	vtkMetaImageReader*			pMetaImageReader		= dynamic_cast<vtkMetaImageReader*>(pCaller);
	vtkImageResample*			pImageResample			= dynamic_cast<vtkImageResample*>(pCaller);
	vtkImageGradientMagnitude*	pImageGradientMagnitude	= dynamic_cast<vtkImageGradientMagnitude*>(pCaller);

	if (gpProgressDialog)
	{
		gpProgressDialog->resize(400, 100);

		if (pMetaImageReader)
		{
			gpProgressDialog->setLabelText("Loading volume");
			gpProgressDialog->setValue((int)(pMetaImageReader->GetProgress() * 100.0));
		}

		if (pImageResample)
		{
			gpProgressDialog->setLabelText("Resampling volume");
			gpProgressDialog->setValue((int)(pImageResample->GetProgress() * 100.0));
		}

		if (pImageGradientMagnitude)
		{
			gpProgressDialog->setLabelText("Creating gradient magnitude volume");
			gpProgressDialog->setValue((int)(pImageGradientMagnitude->GetProgress() * 100.0));
		}
	}
}

CRenderThread::CRenderThread(QObject* pParent) :
	QThread(pParent),
	m_FileName(),
	m_N(0),
	m_pRenderImage(NULL),
	m_pImageDataVolume(NULL),
	m_Scene(),
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
	// Initialize CUDA and set the device
	if (!InitializeCuda())
	{
		QMessageBox::critical(gpMainWindow, "An error has occured", "Unable to locate a CUDA capable device, with streaming architecture 1.1 or higher");
		return;
	}

	m_Scene.m_Camera.m_Film.m_Resolution.Set(Vec2i(800, 600));
	m_Scene.m_Camera.m_Aperture.m_Size = 0.01f;
	m_Scene.m_Camera.m_Focus.m_FocalDistance = (m_Scene.m_Camera.m_Target - m_Scene.m_Camera.m_From).Length();
	m_Scene.m_Camera.m_SceneBoundingBox = m_Scene.m_BoundingBox;
	m_Scene.m_Camera.SetViewMode(ViewModeFront);
	m_Scene.m_Camera.Update();

	// Force the render thread to allocate the necessary buffers, do not remove this line
	m_Scene.m_DirtyFlags.SetFlag(FilmResolutionDirty | CameraDirty);

//	short* pData = (short*)malloc(m_Scene.m_Resolution.m_NoElements * sizeof(short));
	
	// Bind the volume
	BindVolumeData((short*)m_pImageDataVolume->GetScalarPointer(), m_Scene.m_Resolution);
//	BindVolumeData(pData, m_Scene.m_Resolution);
	
//	free(pData);

	// Allocate CUDA memory for scene
	cudaMalloc((void**)&m_pDevScene, sizeof(CScene));

	// Let others know that we are starting with rendering
	emit gRenderStatus.RenderBegin();
	
	while (!m_Abort)
	{
		// Let others know we are starting with a new frame
		emit gRenderStatus.PreRenderFrame();

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
		emit gRenderStatus.PostRenderFrame();
	}

	// Let others know that we have stopped rendering
	emit gRenderStatus.RenderEnd();

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

void CRenderThread::Close(void)
{
	qDebug("Closing render thread");
	m_Abort = true;
}

QString CRenderThread::GetFileName(void) const
{
	return m_FileName;
}

int CRenderThread::GetNoIterations(void) const
{
	return m_N;
}

unsigned char* CRenderThread::GetRenderImage(void) const
{
	return m_pRenderImage;
}

CScene* CRenderThread::GetScene(void)
{
	return &m_Scene;
}

bool CRenderThread::Load(QString& FileName)
{
	m_FileName = FileName;

//	qDebug(QString("Loading volume " + QFileInfo(m_FileName).fileName()).toAscii());
//	CLoadSettingsDialog LoadSettingsDialog;

	// Make it a modal dialog
//	LoadSettingsDialog.setWindowModality(Qt::WindowModal);
	 
	// Show it
//	LoadSettingsDialog.exec();

	// Create and configure progress dialog
//	gpProgressDialog = new QProgressDialog("Volume loading in progress", "Abort", 0, 100);
//	gpProgressDialog->setWindowTitle("Progress");
//	gpProgressDialog->setMinimumDuration(10);
//	gpProgressDialog->setWindowFlags(Qt::Popup);
//	gpProgressDialog->show();

	// Create meta image reader
	vtkSmartPointer<vtkMetaImageReader> MetaImageReader = vtkMetaImageReader::New();

	// Exit if the reader can't read the file
	if (!MetaImageReader->CanReadFile(m_FileName.toAscii()))
		return false;

	// Create progress callback
	vtkSmartPointer<vtkCallbackCommand> ProgressCallback = vtkSmartPointer<vtkCallbackCommand>::New();

	// Set callback
	ProgressCallback->SetCallback (OnProgress);
	ProgressCallback->SetClientData(MetaImageReader);

	// Progress handling
//	MetaImageReader->AddObserver(vtkCommand::ProgressEvent, ProgressCallback);

	MetaImageReader->SetFileName(m_FileName.toAscii());

	MetaImageReader->Update();

	vtkSmartPointer<vtkImageCast> ImageCast = vtkImageCast::New();

	ImageCast->SetOutputScalarTypeToShort();
	ImageCast->SetInput(MetaImageReader->GetOutput());
	
	ImageCast->Update();

	m_pImageDataVolume = ImageCast->GetOutput();

	/*
//	if (LoadSettingsDialog.GetResample())
//	{
		// Create resampler
		vtkSmartPointer<vtkImageResample> ImageResample = vtkImageResample::New();

		// Progress handling
		ImageResample->AddObserver(vtkCommand::ProgressEvent, ProgressCallback);

		ImageResample->SetInput(m_pImageDataVolume);

		// Obtain resampling scales from dialog input
		gm_Scene.m_Scale.x = LoadSettingsDialog.GetResampleX();
		gm_Scene.m_Scale.y = LoadSettingsDialog.GetResampleY();
		gm_Scene.m_Scale.z = LoadSettingsDialog.GetResampleZ();

		// Apply scaling factors
		ImageResample->SetAxisMagnificationFactor(0, gm_Scene.m_Scale.x);
		ImageResample->SetAxisMagnificationFactor(1, gm_Scene.m_Scale.y);
		ImageResample->SetAxisMagnificationFactor(2, gm_Scene.m_Scale.z);
	
		// Resample
		ImageResample->Update();

		m_pImageDataVolume = ImageResample->GetOutput();
//	}
	*/
	/*
	// Create magnitude volume
	vtkSmartPointer<vtkImageGradientMagnitude> ImageGradientMagnitude = vtkImageGradientMagnitude::New();
	
	// Progress handling
	ImageGradientMagnitude->AddObserver(vtkCommand::ProgressEvent, ProgressCallback);
	
	ImageGradientMagnitude->SetInput(pImageData);
	ImageGradientMagnitude->Update();
	*/

	

	m_Scene.m_MemorySize	= (float)m_pImageDataVolume->GetActualMemorySize() / 1024.0f;
	

	double Range[2];

	m_pImageDataVolume->GetScalarRange(Range);

	m_Scene.m_IntensityRange.m_Min	= (float)Range[0];
	m_Scene.m_IntensityRange.m_Max	= (float)Range[1];

	gTransferFunction.SetRangeMin((float)Range[0]);
	gTransferFunction.SetRangeMax((float)Range[1]);

	int* pExtent = m_pImageDataVolume->GetExtent();
	
	m_Scene.m_Resolution.m_XYZ.x = pExtent[1] + 1;
	m_Scene.m_Resolution.m_XYZ.y = pExtent[3] + 1;
	m_Scene.m_Resolution.m_XYZ.z = pExtent[5] + 1;
	m_Scene.m_Resolution.Update();

	double* pSpacing = m_pImageDataVolume->GetSpacing();

	
	m_Scene.m_Spacing.x = pSpacing[0];
	m_Scene.m_Spacing.y = pSpacing[1];
	m_Scene.m_Spacing.z = pSpacing[2];
	

	Vec3f Resolution = Vec3f(m_Scene.m_Spacing.x * (float)m_Scene.m_Resolution.m_XYZ.x, m_Scene.m_Spacing.y * (float)m_Scene.m_Resolution.m_XYZ.y, m_Scene.m_Spacing.z * (float)m_Scene.m_Resolution.m_XYZ.z);

	float Max = Resolution.Max();

	m_Scene.m_NoVoxels				= m_Scene.m_Resolution.m_NoElements;
	m_Scene.m_BoundingBox.m_MinP	= Vec3f(0.0f);
	m_Scene.m_BoundingBox.m_MaxP	= Vec3f(Resolution.x / Max, Resolution.y / Max, Resolution.z / Max);

	// Build the histogram
	vtkSmartPointer<vtkImageAccumulate> Histogram = vtkSmartPointer<vtkImageAccumulate>::New();
// 	Histogram->SetInputConnection(ImageResample->GetOutputPort());
// 	Histogram->SetComponentExtent(0, 1024, 0, 0, 0, 0);
// 	Histogram->SetComponentOrigin(0, 0, 0);
// 	Histogram->SetComponentSpacing(1, 0, 0);
// 	Histogram->IgnoreZeroOn();
// 	Histogram->Update();
 
	// Update the histogram in the transfer function
//	gTransferFunction.SetHistogram((int*)Histogram->GetOutput()->GetScalarPointer(), 256);
	
	// Delete progress dialog
//	gpProgressDialog->close();
//	delete gpProgressDialog;
//	gpProgressDialog = NULL;

	return true;
}

CScene* Scene(void)
{
	if (gpRenderThread)
		return gpRenderThread->GetScene();

	return NULL;
}

void StartRenderThread(QString& FileName)
{
	// Create new render thread
	gpRenderThread = new CRenderThread(NULL);

	// Load the volume
	gpRenderThread->Load(FileName);

	// Start the render thread
	gpRenderThread->start();
}

void KillRenderThread(void)
{
	if (!gpRenderThread)
		return;

	// Kill the render thread
	gpRenderThread->Close();

	// Wait for thread to end
	gpRenderThread->wait();

	// Remove the render thread
	delete gpRenderThread;
	gpRenderThread = NULL;

	qDebug("Render thread killed");
}
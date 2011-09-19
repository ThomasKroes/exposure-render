
// Precompiled headers
#include "Stable.h"

#include "RenderThread.h"
#include "Scene.h"
#include "MainWindow.h"
#include "LoadSettingsDialog.h"

// CUDA kernels
#include "Core.cuh"

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
QRenderThread* gpRenderThread = NULL;

// This function wraps the CUDA Driver API into a template function
template <class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute, int device)
{
	CUresult error = 	cuDeviceGetAttribute( attribute, device_attribute, device );

	if( CUDA_SUCCESS != error) {
		fprintf(stderr, "cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n",
			error, __FILE__, __LINE__);
		exit(-1);
	}
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

// 	if (gpProgressDialog)
// 	{
// 		gpProgressDialog->resize(400, 100);

		if (pMetaImageReader)
		{
//			LogProgress("Loading Volume", (float)pMetaImageReader->GetProgress() * 100.0f);
//			gpProgressDialog->setLabelText("Loading volume");
//			gpProgressDialog->setValue((int)(pMetaImageReader->GetProgress() * 100.0));
		}

		if (pImageResample)
		{
//			gpProgressDialog->setLabelText("Resampling volume");
//			gpProgressDialog->setValue((int)(pImageResample->GetProgress() * 100.0));
		}

		if (pImageGradientMagnitude)
		{
//			gpProgressDialog->setLabelText("Creating gradient magnitude volume");
//			gpProgressDialog->setValue((int)(pImageGradientMagnitude->GetProgress() * 100.0));
		}
//	}
}

QString FormatVector(const Vec3f& Vector, const int& Precision = 2)
{
	return "[" + QString::number(Vector.x, 'f', Precision) + ", " + QString::number(Vector.y, 'f', Precision) + ", " + QString::number(Vector.z, 'f', Precision) + "]";
}

QString FormatVector(const Vec3i& Vector)
{
	return "[" + QString::number(Vector.x) + ", " + QString::number(Vector.y) + ", " + QString::number(Vector.z) + "]";
}

QString FormatSize(const Vec3f& Size, const int& Precision = 2)
{
	return QString::number(Size.x, 'f', Precision) + " x " + QString::number(Size.y, 'f', Precision) + " x " + QString::number(Size.z, 'f', Precision);
}

QString FormatSize(const Vec3i& Size)
{
	return QString::number(Size.x) + " x " + QString::number(Size.y) + " x " + QString::number(Size.z);
}

QRenderThread::QRenderThread(const QString& FileName, QObject* pParent /*= NULL*/) :
	QThread(pParent),
	m_Mutex(),
	m_FileName(FileName),
	m_N(0),
	m_pRenderImage(NULL),
	m_pImageDataVolume(NULL),
// 	m_Scene(),
	m_pScene(NULL),
	m_pDevAccEstXyz(NULL),
	m_pDevEstFrameXyz(NULL),
	m_pDevEstFrameBlurXyz(NULL),
	m_pDevEstRgbLdr(NULL),
	m_pDevRgbLdrDisp(NULL),
	m_pImageCanvas(NULL),
	m_pSeeds(NULL),
	m_Abort(false)
{
}

QRenderThread::QRenderThread(const QRenderThread& Other)
{
	*this = Other;
}

QRenderThread& QRenderThread::operator=(const QRenderThread& Other)
{
// 	m_Mutex					= Other.m_Mutex;
	m_FileName				= Other.m_FileName;
	m_pRenderImage			= Other.m_pRenderImage;
	m_pRenderImage			= Other.m_pRenderImage;
	m_pImageDataVolume		= Other.m_pImageDataVolume;
	m_Scene					= Other.m_Scene;
	m_pScene				= Other.m_pScene;
	m_pDevAccEstXyz			= Other.m_pDevAccEstXyz;
	m_pDevEstFrameXyz		= Other.m_pDevEstFrameXyz;
	m_pDevEstFrameBlurXyz	= Other.m_pDevEstFrameBlurXyz;
	m_pDevEstRgbLdr			= Other.m_pDevEstRgbLdr;
	m_pDevRgbLdrDisp		= Other.m_pDevRgbLdrDisp;
	m_pImageCanvas			= Other.m_pImageCanvas;
	m_Abort					= Other.m_Abort;
	m_pSeeds				= m_pSeeds;

	return *this;
}

QRenderThread::~QRenderThread(void)
{
	Log("Render thread destroyed");
}

bool QRenderThread::InitializeCuda(void)
{
	// No CUDA enabled devices
	int NoDevices = 0;

	cudaError_t ErrorID = cudaGetDeviceCount(&NoDevices);


	emit gRenderStatus.StatisticChanged("Graphics Card", "No. CUDA capable devices", QString::number(NoDevices));

	Log("Found " + QString::number(NoDevices) + " CUDA enabled device(s)", "graphic-card");

	int DriverVersion = 0, RuntimeVersion = 0; 

	cudaDriverGetVersion(&DriverVersion);
	cudaRuntimeGetVersion(&RuntimeVersion);

	QString DriverVersionString		= QString::number(DriverVersion / 1000) + "." + QString::number(DriverVersion % 100);
	QString RuntimeVersionString	= QString::number(RuntimeVersion / 1000) + "." + QString::number(RuntimeVersion % 100);

	emit gRenderStatus.StatisticChanged("Graphics Card", "CUDA Driver Version", DriverVersionString);
	emit gRenderStatus.StatisticChanged("Graphics Card", "CUDA Runtime Version", RuntimeVersionString);

	Log("Driver version " + DriverVersionString, "graphic-card");
	Log("Runtime version " + RuntimeVersionString, "graphic-card");

	for (int Device = 0; Device < NoDevices; Device++)
	{
		QString DeviceString = "Device " + QString::number(Device);

		emit gRenderStatus.StatisticChanged("Graphics Card", DeviceString, "");

		cudaDeviceProp DeviceProperties;
		cudaGetDeviceProperties(&DeviceProperties, Device);

		QString CudaCapabilityString = QString::number(DeviceProperties.major) + "." + QString::number(DeviceProperties.minor);
		
		emit gRenderStatus.StatisticChanged(DeviceString, "CUDA Capability", CudaCapabilityString);

		// Memory
		emit gRenderStatus.StatisticChanged(DeviceString, "On Board Memory", "", "", "memory");

		emit gRenderStatus.StatisticChanged("On Board Memory", "Total Global Memory", QString::number((float)DeviceProperties.totalGlobalMem / powf(1024.0f, 2.0f), 'f', 2), "MB");
		emit gRenderStatus.StatisticChanged("On Board Memory", "Total Constant Memory", QString::number((float)DeviceProperties.totalConstMem / powf(1024.0f, 2.0f), 'f', 2), "MB");

		int MemoryClock, MemoryBusWidth, L2CacheSize;
		getCudaAttribute<int>(&MemoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, Device);
		getCudaAttribute<int>(&MemoryBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, Device);
		getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, Device);

		emit gRenderStatus.StatisticChanged("On Board Memory", "Memory Clock Rate", QString::number(MemoryClock * 1e-3f), "Mhz");
		emit gRenderStatus.StatisticChanged("On Board Memory", "Memory Bus Width", QString::number(MemoryBusWidth), "bit");
		emit gRenderStatus.StatisticChanged("On Board Memory", "L2 Cache Size", QString::number(L2CacheSize), "bytes");
		emit gRenderStatus.StatisticChanged("On Board Memory", "Maximum Memory Pitch", QString::number((float)DeviceProperties.memPitch / powf(1024.0f, 2.0f), 'f', 2), "MB");
		
		// Processor
		emit gRenderStatus.StatisticChanged(DeviceString, "Processor", "", "", "processor");
		emit gRenderStatus.StatisticChanged("Processor", "No. Multiprocessors", QString::number(DeviceProperties.multiProcessorCount), "Processors");
		emit gRenderStatus.StatisticChanged("Processor", "GPU Clock Speed", QString::number(DeviceProperties.clockRate * 1e-6f, 'f', 2), "GHz");
		emit gRenderStatus.StatisticChanged("Processor", "Max. Block Size", QString::number(DeviceProperties.maxThreadsDim[0]) + " x " + QString::number(DeviceProperties.maxThreadsDim[1]) + " x " + QString::number(DeviceProperties.maxThreadsDim[2]), "Threads");
		emit gRenderStatus.StatisticChanged("Processor", "Max. Grid Size", QString::number(DeviceProperties.maxGridSize[0]) + " x " + QString::number(DeviceProperties.maxGridSize[1]) + " x " + QString::number(DeviceProperties.maxGridSize[2]), "Blocks");
		emit gRenderStatus.StatisticChanged("Processor", "Warp Size", QString::number(DeviceProperties.warpSize), "Threads");
		emit gRenderStatus.StatisticChanged("Processor", "Max. No. Threads/Block", QString::number(DeviceProperties.maxThreadsPerBlock), "Threads");
		emit gRenderStatus.StatisticChanged("Processor", "Max. Shared Memory Per Block", QString::number((float)DeviceProperties.sharedMemPerBlock / 1024.0f, 'f', 2), "KB");
		emit gRenderStatus.StatisticChanged("Processor", "Registers Available Per Block", QString::number((float)DeviceProperties.regsPerBlock / 1024.0f, 'f', 2), "KB");

		// Texture
		emit gRenderStatus.StatisticChanged(DeviceString, "Texture", "", "", "checkerboard");
		emit gRenderStatus.StatisticChanged("Texture", "Max. Dimension Size 1D", QString::number(DeviceProperties.maxTexture1D), "Pixels");
		emit gRenderStatus.StatisticChanged("Texture", "Max. Dimension Size 2D", QString::number(DeviceProperties.maxTexture2D[0]) + " x " + QString::number(DeviceProperties.maxTexture2D[1]), "Pixels");
		emit gRenderStatus.StatisticChanged("Texture", "Max. Dimension Size 3D", QString::number(DeviceProperties.maxTexture3D[0]) + " x " + QString::number(DeviceProperties.maxTexture3D[1]) + " x " + QString::number(DeviceProperties.maxTexture3D[2]), "Pixels");
		emit gRenderStatus.StatisticChanged("Texture", "Alignment", QString::number((float)DeviceProperties.textureAlignment / powf(1024.0f, 2.0f), 'f', 2), "MB");
	}	
	
	return cudaSetDevice(cutGetMaxGflopsDeviceId()) == cudaSuccess;
}

void QRenderThread::run()
{
 	if (!InitializeCuda())
 	{
		Log("Unable to initialize CUDA, rendering cannot start", QLogger::Critical);
 		QMessageBox::critical(gpMainWindow, "An error has occurred", "Unable to locate a CUDA capable device");
 		return;
 	}

	m_Scene.m_Camera.m_Film.m_Resolution.Set(Vec2i(640, 480));
 	m_Scene.m_Camera.m_SceneBoundingBox = m_Scene.m_BoundingBox;
 	m_Scene.m_Camera.SetViewMode(ViewModeFront);
 	m_Scene.m_Camera.Update();

	// Force the render thread to allocate the necessary buffers, do not remove this line
	m_Scene.m_DirtyFlags.SetFlag(FilmResolutionDirty | CameraDirty);

 	CreateVolume();

	emit gRenderStatus.StatisticChanged("Performance", "Timings", "");

	// Allocate CUDA memory for scene
	HandleCudaError(cudaMalloc((void**)&m_pScene, sizeof(CScene)));

	emit gRenderStatus.StatisticChanged("CUDA", "Scene", QString::number(sizeof(CScene) / powf(1024.0f, 2.0f), 'f', 2), "MB");

	// Let others know that we are starting with rendering
	emit gRenderStatus.RenderBegin();
	
	// Try to load appearance/lighting/camera presets with the same name as the loaded file
	emit gRenderStatus.LoadPreset(QFileInfo(m_FileName).baseName());

	// Keep track of frames/second
	CEvent FPS;

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

		emit gRenderStatus.StatisticChanged("Camera", "Position", FormatVector(m_Scene.m_Camera.m_From));
		emit gRenderStatus.StatisticChanged("Camera", "Target", FormatVector(m_Scene.m_Camera.m_Target));
		emit gRenderStatus.StatisticChanged("Camera", "Up Vector", FormatVector(m_Scene.m_Camera.m_Up));
		emit gRenderStatus.StatisticChanged("Camera", "Aperture Size", QString::number(Scene()->m_Camera.m_Aperture.m_Size, 'f', 2));
		emit gRenderStatus.StatisticChanged("Camera", "Field Of View", QString::number(Scene()->m_Camera.m_FovV, 'f', 2));

		// Copy scene from host to device
		cudaMemcpy(m_pScene, &SceneCopy, sizeof(CScene), cudaMemcpyHostToDevice);

		// Resizing the image canvas requires special attention
		if (SceneCopy.m_DirtyFlags.HasFlag(FilmResolutionDirty))
		{
			m_Mutex.lock();

			// Allocate host image buffer, this thread will blit it's frames to this buffer
			free(m_pRenderImage);
			m_pRenderImage = NULL;

			m_pRenderImage = (unsigned char*)malloc(3 * SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(unsigned char));

			if (m_pRenderImage)
				memset(m_pRenderImage, 0, 3 * SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(unsigned char));
			
			emit gRenderStatus.StatisticChanged("Host", "LDR Frame Buffer", QString::number(3 * SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(unsigned char) / powf(1024.0f, 2.0f), 'f', 2), "MB");

			m_Mutex.unlock();

			const int NoRandomSeeds = SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * 2;

			// Compute size of the CUDA buffers
 			float SizeRandomSeeds			= NoRandomSeeds * sizeof(unsigned int);
			float SizeHdrAccumulationBuffer	= SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorXyz);
			float SizeHdrFrameBuffer		= SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorXyz);
			float SizeHdrBlurFrameBuffer	= SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorXyz);
			float SizeLdrFrameBuffer		= 3 * SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(unsigned char);
			
			HandleCudaError(cudaFree(m_pSeeds));
			HandleCudaError(cudaFree(m_pDevAccEstXyz));
			HandleCudaError(cudaFree(m_pDevEstFrameXyz));
			HandleCudaError(cudaFree(m_pDevEstFrameBlurXyz));
			HandleCudaError(cudaFree(m_pDevEstRgbLdr));
			HandleCudaError(cudaFree(m_pDevRgbLdrDisp));

			// Allocate device buffers
			HandleCudaError(cudaMalloc((void**)&m_pSeeds, SizeRandomSeeds));
			HandleCudaError(cudaMalloc((void**)&m_pDevAccEstXyz, SizeHdrAccumulationBuffer));
			HandleCudaError(cudaMalloc((void**)&m_pDevEstFrameXyz, SizeHdrFrameBuffer));
			HandleCudaError(cudaMalloc((void**)&m_pDevEstFrameBlurXyz, SizeHdrBlurFrameBuffer));
			HandleCudaError(cudaMalloc((void**)&m_pDevEstRgbLdr, SizeLdrFrameBuffer));
			HandleCudaError(cudaMalloc((void**)&m_pDevRgbLdrDisp, SizeLdrFrameBuffer));

// 			emit gRenderStatus.StatisticChanged("CUDA", "Random States", QString::number(m_SizeRandomStates / powf(1024.0f, 2.0f), 'f', 2), "MB", ":/Images/memory.png");
			emit gRenderStatus.StatisticChanged("CUDA", "HDR Accumulation Buffer", QString::number(SizeHdrFrameBuffer / powf(1024.0f, 2.0f), 'f', 2), "MB");
			emit gRenderStatus.StatisticChanged("CUDA", "HDR Frame Buffer Blur", QString::number(SizeHdrBlurFrameBuffer / powf(1024.0f, 2.0f), 'f', 2), "MB");
			emit gRenderStatus.StatisticChanged("CUDA", "LDR Estimation Buffer", QString::number(SizeLdrFrameBuffer / powf(1024.0f, 2.0f), 'f', 2), "MB");
			
			// Create random seeds
			unsigned int* pSeeds = (unsigned int*)malloc(SizeRandomSeeds);

			for (int i = 0; i < NoRandomSeeds; i++)
				pSeeds[i] = qrand();

			HandleCudaError(cudaMemcpy(m_pSeeds, pSeeds, SizeRandomSeeds, cudaMemcpyHostToDevice));

			free(pSeeds);

			// Reset buffers to black
			HandleCudaError(cudaMemset(m_pDevAccEstXyz, 0, SizeHdrAccumulationBuffer));
			HandleCudaError(cudaMemset(m_pDevEstFrameXyz, 0, SizeHdrFrameBuffer));
			HandleCudaError(cudaMemset(m_pDevEstFrameBlurXyz, 0, SizeHdrBlurFrameBuffer));
			HandleCudaError(cudaMemset(m_pDevEstRgbLdr, 0, SizeLdrFrameBuffer));
			HandleCudaError(cudaMemset(m_pDevRgbLdrDisp, 0, SizeLdrFrameBuffer));

			// Reset no. iterations
			m_N = 0.0f;

			// Notify Inform others about the memory allocations
			emit gRenderStatus.Resize();

			emit gRenderStatus.StatisticChanged("Camera", "Resolution", QString::number(SceneCopy.m_Camera.m_Film.m_Resolution.GetResX()) + " x " + QString::number(SceneCopy.m_Camera.m_Film.m_Resolution.GetResY()), "Pixels");
		}

		// Restart the rendering when when the camera, lights and render params are dirty
		if (SceneCopy.m_DirtyFlags.HasFlag(CameraDirty | LightsDirty | RenderParamsDirty | TransferFunctionDirty))
		{
			// Reset buffers to black
			HandleCudaError(cudaMemset(m_pDevAccEstXyz, 0, SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorXyz)));
			HandleCudaError(cudaMemset(m_pDevEstFrameXyz, 0, SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorXyz)));
			HandleCudaError(cudaMemset(m_pDevEstFrameBlurXyz, 0, SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorXyz)));
			HandleCudaError(cudaMemset(m_pDevEstRgbLdr, 0, SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(unsigned char)));
//			HandleCudaError(cudaMemset(m_pDevRgbLdrDisp, 0, SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(unsigned char)));

			// Reset no. iterations
			m_N = 0.0f;
		}

		// At this point, all dirty flags should have been taken care of, since the flags in the original scene are now cleared
		m_Scene.m_DirtyFlags.ClearAllFlags();

		CCudaTimer Timer;

		// Execute the rendering kernels
  		Render(0, &SceneCopy, m_pScene, m_pSeeds, m_pDevEstFrameXyz, m_pDevEstFrameBlurXyz, m_pDevAccEstXyz, m_pDevEstRgbLdr, m_pDevRgbLdrDisp, m_N);
		HandleCudaError(cudaGetLastError());
		
		emit gRenderStatus.StatisticChanged("Timings", "Integration + Tone Mapping", QString::number(Timer.StopTimer(), 'f', 2), "ms");

		// Increase the number of iterations performed so far
		m_N++;

		FPS.AddDuration(1000.0f / CudaTimer.StopTimer());

		emit gRenderStatus.StatisticChanged("Performance", "FPS", QString::number(FPS.m_FilteredDuration, 'f', 2), "Frames/Sec.");
		emit gRenderStatus.StatisticChanged("Performance", "No. Iterations", QString::number(m_N), "Iterations");

		// Blit
		HandleCudaError(cudaMemcpy(m_pRenderImage, m_pDevRgbLdrDisp, 3 * SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(unsigned char), cudaMemcpyDeviceToHost));

		// Let others know we are finished with a frame
		emit gRenderStatus.PostRenderFrame();
	}

	// Let others know that we have stopped rendering
	emit gRenderStatus.RenderEnd();

	// Load default appearance, lighting and camera presets
	emit gRenderStatus.LoadPreset("Default");

	// Free CUDA buffers
	HandleCudaError(cudaFree(m_pScene));
	HandleCudaError(cudaFree(m_pSeeds));
	HandleCudaError(cudaFree(m_pDevAccEstXyz));
	HandleCudaError(cudaFree(m_pDevEstFrameXyz));
	HandleCudaError(cudaFree(m_pDevEstFrameBlurXyz));
	HandleCudaError(cudaFree(m_pDevEstRgbLdr));
	HandleCudaError(cudaFree(m_pDevRgbLdrDisp));
	
	m_pScene				= NULL;
	m_pSeeds				= NULL;
	m_pDevAccEstXyz			= NULL;
	m_pDevEstFrameXyz		= NULL;
	m_pDevEstFrameBlurXyz	= NULL;
	m_pDevEstRgbLdr			= NULL;

	// Free render image buffer
	free(m_pRenderImage);
	m_pRenderImage = NULL;

	// Clear the histogram
	gHistogram.Reset();
}

void QRenderThread::Close(void)
{
	m_Abort = true;
}

QString QRenderThread::GetFileName(void) const
{
	return m_FileName;
}

void QRenderThread::SetFileName(const QString& FileName)
{
	m_FileName = FileName;
}

int QRenderThread::GetNoIterations(void) const
{
	return m_N;
}

unsigned char* QRenderThread::GetRenderImage(void) const
{
	return m_pRenderImage;
}

CScene* QRenderThread::GetScene(void)
{
	return &m_Scene;
}

bool QRenderThread::Load(QString& FileName)
{
	m_FileName = FileName;
 	QLoadSettingsDialog LoadSettingsDialog;
 
 	// Make it a modal dialog
 	LoadSettingsDialog.setWindowModality(Qt::WindowModal);
 	 
 	// Show it
 	LoadSettingsDialog.exec();

	// Create and configure progress dialog
//	gpProgressDialog = new QProgressDialog("Volume loading in progress", "Abort", 0, 100);
//	gpProgressDialog->setWindowTitle("Progress");
//	gpProgressDialog->setMinimumDuration(10);
//	gpProgressDialog->setWindowFlags(Qt::Popup);
//	gpProgressDialog->show();

// 	emit gRenderStatus.StatisticChanged("Memory", "CUDA", "", "", "memory");
// 	emit gRenderStatus.StatisticChanged("Memory", "Host", "", "", "memory");

	// Create meta image reader
	vtkSmartPointer<vtkMetaImageReader> MetaImageReader = vtkMetaImageReader::New();

	QFileInfo FileInfo(FileName);

	if (!FileInfo.exists())
	{
		Log(QString(QFileInfo(FileName).filePath().replace("//", "/")).toAscii() + "  does not exist!", QLogger::Critical);
		return false;
	}

	// Exit if the reader can't read the file
	if (!MetaImageReader->CanReadFile(m_FileName.toAscii()))
	{
		Log(QString("Meta image reader can't read file " + QFileInfo(FileName).fileName()).toAscii(), QLogger::Critical);
		return false;
	}

	// Create progress callback
	vtkSmartPointer<vtkCallbackCommand> ProgressCallback = vtkSmartPointer<vtkCallbackCommand>::New();

	// Set callback
	ProgressCallback->SetCallback (OnProgress);
	ProgressCallback->SetClientData(MetaImageReader);

	// Progress handling
	MetaImageReader->AddObserver(vtkCommand::ProgressEvent, ProgressCallback);

	MetaImageReader->SetFileName(m_FileName.toAscii());

	Log(QString("Loading " + QFileInfo(FileName).fileName()).toAscii());

	MetaImageReader->Update();

	vtkSmartPointer<vtkImageCast> ImageCast = vtkImageCast::New();

//	Log("Casting volume data type to short");

	ImageCast->SetOutputScalarTypeToShort();
	ImageCast->SetInput(MetaImageReader->GetOutput());
	
	ImageCast->Update();

	m_pImageDataVolume = ImageCast->GetOutput();
	
	
// 	if (LoadSettingsDialog.GetResample())
// 	{
		Log("Resampling volume at " + QString::number(LoadSettingsDialog.GetResampleX(), 'f', 2) + " x " + QString::number(LoadSettingsDialog.GetResampleY(), 'f', 2) + " x " + QString::number(LoadSettingsDialog.GetResampleZ(), 'f', 2));

		// Create resampler
		vtkSmartPointer<vtkImageResample> ImageResample = vtkImageResample::New();

		// Progress handling
		ImageResample->AddObserver(vtkCommand::ProgressEvent, ProgressCallback);

		ImageResample->SetInput(m_pImageDataVolume);

		// Obtain resampling scales from dialog input
		m_Scene.m_Scale.x = LoadSettingsDialog.GetResampleX();
		m_Scene.m_Scale.y = LoadSettingsDialog.GetResampleY();
		m_Scene.m_Scale.z = LoadSettingsDialog.GetResampleZ();

		// Apply scaling factors
		ImageResample->SetAxisMagnificationFactor(0, m_Scene.m_Scale.x);
		ImageResample->SetAxisMagnificationFactor(1, m_Scene.m_Scale.y);
		ImageResample->SetAxisMagnificationFactor(2, m_Scene.m_Scale.z);
	
		// Resample
		ImageResample->Update();

		m_pImageDataVolume = ImageResample->GetOutput();
//	}

	/*
	// Create magnitude volume
	vtkSmartPointer<vtkImageGradientMagnitude> ImageGradientMagnitude = vtkImageGradientMagnitude::New();
	
	// Progress handling
	ImageGradientMagnitude->AddObserver(vtkCommand::ProgressEvent, ProgressCallback);
	
	ImageGradientMagnitude->SetInput(pImageData);
	ImageGradientMagnitude->Update();
	*/

//	emit gRenderStatus.StatisticChanged("Host", "Volume", QString::number((float)m_pImageDataVolume->GetActualMemorySize() / 1024.0f, 'f', 2), "MB");


	// Scalar range
	double* pRange = m_pImageDataVolume->GetScalarRange();
	m_Scene.m_IntensityRange.m_Min	= (float)pRange[0];
	m_Scene.m_IntensityRange.m_Max	= (float)pRange[1];

	// Get extent
	int* pExtent = m_pImageDataVolume->GetExtent();
	m_Scene.m_Resolution.SetResXYZ(Vec3i(pExtent[1] + 1, pExtent[3] + 1, pExtent[5] + 1));

	// Spacing
	double* pSpacing = m_pImageDataVolume->GetSpacing();
	
	// Voxel spacing is typically in mm exposure render work in meters so convert to meters
	m_Scene.m_Spacing.x = (float)pSpacing[0];
	m_Scene.m_Spacing.y = (float)pSpacing[1];
	m_Scene.m_Spacing.z = (float)pSpacing[2];

	// Compute physical size
	const Vec3f PhysicalSize(Vec3f(m_Scene.m_Spacing.x * (float)m_Scene.m_Resolution.GetResX(), m_Scene.m_Spacing.y * (float)m_Scene.m_Resolution.GetResY(), m_Scene.m_Spacing.z * (float)m_Scene.m_Resolution.GetResZ()));

	// Compute the volume's bounding box
	m_Scene.m_BoundingBox.m_MinP	= Vec3f(0.0f);
	m_Scene.m_BoundingBox.m_MaxP	= PhysicalSize / PhysicalSize.Max();

	// Build the histogram
	vtkSmartPointer<vtkImageAccumulate> Histogram = vtkSmartPointer<vtkImageAccumulate>::New();
 	Histogram->SetInputConnection(ImageCast->GetOutputPort());
 	Histogram->SetComponentExtent(0, 2048, 0, 0, 0, 0);
 	Histogram->SetComponentOrigin(0, 0, 0);
 	Histogram->SetComponentSpacing(1, 0, 0);
 	Histogram->IgnoreZeroOn();
 	Histogram->Update();
 
	// Update the histogram in the transfer function
	gHistogram.SetBins((int*)Histogram->GetOutput()->GetScalarPointer(), 2048);
	
	// Delete progress dialog
//	gpProgressDialog->close();
//	delete gpProgressDialog;
//	gpProgressDialog = NULL;

	emit gRenderStatus.StatisticChanged("Volume", "File", QFileInfo(m_FileName).fileName(), "");
	emit gRenderStatus.StatisticChanged("Volume", "Bounding Box", "", "");
	emit gRenderStatus.StatisticChanged("Bounding Box", "Min", FormatVector(m_Scene.m_BoundingBox.m_MinP, 2), "m");
	emit gRenderStatus.StatisticChanged("Bounding Box", "Max", FormatVector(m_Scene.m_BoundingBox.m_MaxP, 2), "m");
	emit gRenderStatus.StatisticChanged("Volume", "Physical Size", FormatSize(PhysicalSize, 2), "mm");
	emit gRenderStatus.StatisticChanged("Volume", "Resolution", FormatSize(m_Scene.m_Resolution.GetResXYZ()), "Voxels");
	emit gRenderStatus.StatisticChanged("Volume", "Spacing", FormatSize(m_Scene.m_Spacing, 2), "mm");
//	emit gRenderStatus.StatisticChanged("Volume", "Scale", FormatVector(1000.0f * m_Scene.m_Scale, 2), "");
	emit gRenderStatus.StatisticChanged("Volume", "No. Voxels", QString::number(m_Scene.m_Resolution.GetNoElements()), "Voxels");
	emit gRenderStatus.StatisticChanged("Volume", "Density Range", "[" + QString::number(m_Scene.m_IntensityRange.m_Min) + ", " + QString::number(m_Scene.m_IntensityRange.m_Max) + "]", "");

	Log("Bounding box: " + FormatVector(m_Scene.m_BoundingBox.m_MinP, 2) + " - " + FormatVector(m_Scene.m_BoundingBox.m_MaxP), "grid");
	Log("Spacing: " + FormatSize(m_Scene.m_Spacing, 2), "grid");
	Log("Resolution after re-sampling: " + FormatSize(m_Scene.m_Resolution.GetResXYZ()) + " mm", "grid");
	Log("Density range: [" + QString::number(m_Scene.m_IntensityRange.m_Min) + ", " + QString::number(m_Scene.m_IntensityRange.m_Max) + "]", "grid");

	// Print scene data
//	m_Scene.PrintSelf();

	return true;
}

void QRenderThread::CreateVolume(void)
{
	Log("Creating density volume", "grid");
	
	cudaExtent DensityBufferSize = make_cudaExtent(m_Scene.m_Resolution[0], m_Scene.m_Resolution[1], m_Scene.m_Resolution[2]);

	// Allocate density buffer
	short* pDensityBuffer = (short*)malloc(m_Scene.m_Resolution.GetNoElements() * sizeof(short));

	// Copy density data from vtk image to density buffer
	m_pImageDataVolume->Update();
	memcpy(pDensityBuffer, m_pImageDataVolume->GetScalarPointer(), m_Scene.m_Resolution.GetNoElements() * sizeof(short));

	m_Scene.m_SigmaMax = 0.0f;

	for (int i = 0; i < m_Scene.m_Resolution.GetNoElements(); ++i)
	{
		m_Scene.m_SigmaMax = m_Scene.m_SigmaMax > pDensityBuffer[i] ? m_Scene.m_SigmaMax : pDensityBuffer[i];
	}

	for (int i = 0; i < m_Scene.m_Resolution.GetNoElements(); ++i)
	{
//		pDensityBuffer[i] = pDensityBuffer[i] > 10.0f ? pDensityBuffer[i] / m_Scene.m_SigmaMax : 0;
//		pDensityBuffer[i] /= m_Scene.m_SigmaMax;
	}

	// Copy to graphics device
	BindDensityVolume(pDensityBuffer, DensityBufferSize);

	// Create the extinction volume
// 	CreateExtinctionVolume(pDensityBuffer, DensityBufferSize);

	free(pDensityBuffer);
}

void QRenderThread::CreateExtinctionVolume(float* pDensityBuffer, cudaExtent densityBufferSize)
{
	Log("Creating extinction volume");

	cudaExtent extinctionSize;

	int macrocellSize = 8;

	extinctionSize.width = densityBufferSize.width/macrocellSize;
	extinctionSize.height = densityBufferSize.height/macrocellSize;
	extinctionSize.depth = densityBufferSize.depth/macrocellSize;

	float* extinction = (float*)malloc(sizeof(float)*extinctionSize.width*extinctionSize.height*extinctionSize.depth);
	for(int i = 0; i<extinctionSize.width*extinctionSize.height*extinctionSize.depth; ++i){
		extinction[i] = 0.0f;
	}

	for(int x = 0; x<densityBufferSize.width; ++x){
		for(int y = 0; y<densityBufferSize.height; ++y){
			for(int z = 0; z<densityBufferSize.depth; ++z){
				int index =
					x / macrocellSize +
					y / macrocellSize * extinctionSize.width +
					z / macrocellSize * extinctionSize.width * extinctionSize.height;

				if( extinction[index] < pDensityBuffer[x + y * densityBufferSize.width + z * densityBufferSize.width * densityBufferSize.height] ){
					extinction[index] = pDensityBuffer[x + y * densityBufferSize.width + z * densityBufferSize.width * densityBufferSize.height];
				}
			}
		}
	}

	BindExtinctionVolume(extinction, extinctionSize);

// 	free(extinction);
}

void QRenderThread::HandleCudaError(const cudaError_t CudaError)
{
	if (CudaError == cudaSuccess)
		return;

	Log(cudaGetErrorString(CudaError));
}

CScene* Scene(void)
{
	return gpRenderThread ? gpRenderThread->GetScene() : NULL;
}

void StartRenderThread(QString& FileName)
{
	// Create new render thread
 	gpRenderThread = new QRenderThread(FileName);
//  	gpRenderThread->setStackSize(150000000);

	// Load the volume
	if (!gpRenderThread->Load(FileName))
		return;

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
}

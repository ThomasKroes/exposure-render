
// Precompiled headers
#include "Stable.h"

#include "RenderThread.h"
#include "CudaUtilities.h"
#include "MainWindow.h"
#include "LoadSettingsDialog.h"
#include "Lighting.h"
#include "Timing.h"

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
#include <vtkImageShiftScale.h>
#include <vtkErrorCode.h>
#include <vtkImageGradient.h>
#include <vtkExtractVectorComponents.h>

// Render thread
QRenderThread* gpRenderThread = NULL;
QFrameBuffer gFrameBuffer;

QMutex gSceneMutex;

QFrameBuffer::QFrameBuffer(void) :
	m_pPixels(NULL),
	m_Width(0),
	m_Height(0),
	m_NoPixels(0),
	m_Mutex()
{
}

QFrameBuffer::QFrameBuffer(const QFrameBuffer& Other)
{
	*this = Other;
}

QFrameBuffer& QFrameBuffer::operator=(const QFrameBuffer& Other)
{
	const bool Dirty = m_Width != Other.m_Width || m_Height != Other.m_Height;

	m_Width		= Other.m_Width;
	m_Height	= Other.m_Height;
	m_NoPixels	= Other.m_NoPixels;

	if (Other.m_pPixels != NULL)
	{
		const int Size = 3 * m_NoPixels * sizeof(unsigned char);

		if (Dirty)
		{
			free(m_pPixels);
			m_pPixels = (unsigned char*)malloc(Size);
		}

		memcpy(m_pPixels, Other.m_pPixels, Size); 
	}
	else
	{
		m_pPixels = NULL;
	}

	return *this;
}

QFrameBuffer::~QFrameBuffer(void)
{
	free(m_pPixels);
}

void QFrameBuffer::Set(unsigned char* pPixels, const int& Width, const int& Height)
{
	const bool Dirty = Width != m_Width || Height != m_Height;

	m_Width		= Width;
	m_Height	= Height;
	m_NoPixels	= m_Width * m_Height;

	if (m_NoPixels <= 0)
		return;

	const int Size = 3 * m_NoPixels * sizeof(unsigned char);

	if (Dirty)
	{
		free(m_pPixels);
		m_pPixels = (unsigned char*)malloc(Size);
	}

	memcpy(m_pPixels, pPixels, Size); 
}

QRenderThread::QRenderThread(const QString& FileName, QObject* pParent /*= NULL*/) :
	QThread(pParent),
	m_FileName(FileName),
	m_pRenderImage(NULL),
	m_pDensityBuffer(NULL),
	m_pGradientMagnitudeBuffer(NULL),
	m_Abort(false),
	m_Pause(false),
	m_SaveFrames(),
	m_SaveBaseName("phase_function")
{
//	m_SaveFrames << 0 << 100 << 200;
}

QRenderThread::QRenderThread(const QRenderThread& Other)
{
	*this = Other;
}

QRenderThread::~QRenderThread(void)
{
	free(m_pDensityBuffer);
}

QRenderThread& QRenderThread::operator=(const QRenderThread& Other)
{
	m_FileName					= Other.m_FileName;
	m_pRenderImage				= Other.m_pRenderImage;
	m_pDensityBuffer			= Other.m_pDensityBuffer;
	m_pGradientMagnitudeBuffer	= Other.m_pGradientMagnitudeBuffer;
	m_Abort						= Other.m_Abort;
	m_Pause						= Other.m_Pause;
	m_SaveFrames				= Other.m_SaveFrames;
	m_SaveBaseName				= Other.m_SaveBaseName;

	return *this;
}

void QRenderThread::run()
{
	Log("Initializing CUDA...", "graphic-card");

// 	if (!InitializeCuda())
//	{
//		Log("Unable to initialize CUDA, rendering cannot start", QLogger::Critical);
 //		QMessageBox::critical(gpMainWindow, "An error has occurred", "Unable to locate a CUDA capable device");
 //		return;
 //	}

	cudaSetDevice(0);

	CScene SceneCopy;
	
 	gScene.m_Camera.m_SceneBoundingBox = gScene.m_BoundingBox;
	gScene.m_Camera.SetViewMode(ViewModeFront);
 	gScene.m_Camera.Update();

	// Force the render thread to allocate the necessary buffers, do not remove this line
	gScene.m_DirtyFlags.SetFlag(FilmResolutionDirty | CameraDirty);

	gStatus.SetStatisticChanged("Memory", "CUDA Memory", "", "", "memory");
	gStatus.SetStatisticChanged("Memory", "Host Memory", "", "", "memory");

	cudaExtent Res;
	Res.width = gScene.m_Resolution[0];
	Res.height = gScene.m_Resolution[1];
	Res.depth = gScene.m_Resolution[2];

	// Bind density buffer to texture
	Log("Copying density volume to device", "grid");
	gStatus.SetStatisticChanged("CUDA Memory", "Density Buffer", QString::number(gScene.m_Resolution.GetNoElements() * sizeof(short) / MB, 'f', 2), "MB");
	BindDensityBuffer((short*)m_pDensityBuffer, Res);

	// Bind gradient magnitude buffer to texture
	Log("Copying gradient magnitude to device", "grid");
	gStatus.SetStatisticChanged("CUDA Memory", "Gradient Magnitude Buffer", QString::number(gScene.m_Resolution.GetNoElements() * sizeof(short) / MB, 'f', 2), "MB");
	BindGradientMagnitudeBuffer((short*)m_pGradientMagnitudeBuffer, Res);

	gStatus.SetStatisticChanged("Performance", "Timings", "");

	gStatus.SetStatisticChanged("CUDA Memory", "Scene", QString::number(sizeof(CScene) / MB, 'f', 2), "MB");
	gStatus.SetStatisticChanged("CUDA Memory", "Frame Buffers", "", "");

	// Let others know that we are starting with rendering
	gStatus.SetRenderBegin();
	
	Log("Device memory: " + QString::number(GetUsedCudaMemory() / MB, 'f', 2) + "/" + QString::number(GetTotalCudaMemory() / MB, 'f', 2) + " MB", "memory");

	QObject::connect(&gTransferFunction, SIGNAL(Changed()), this, SLOT(OnUpdateTransferFunction()));
	QObject::connect(&gCamera, SIGNAL(Changed()), this, SLOT(OnUpdateCamera()));
	QObject::connect(&gLighting, SIGNAL(Changed()), this, SLOT(OnUpdateLighting()));
	QObject::connect(&gLighting.Background(), SIGNAL(Changed()), this, SLOT(OnUpdateLighting()));

	QObject::connect(&gStatus, SIGNAL(RenderPause(const bool&)), this, SLOT(OnRenderPause(const bool&)));

	// Try to load appearance/lighting/camera presets with the same name as the loaded file
	gStatus.SetLoadPreset(QFileInfo(m_FileName).baseName());

	// Keep track of frames/second
	CTiming FPS, RenderImage, BlurImage, PostProcessImage, DenoiseImage;

	try
	{
		while (!m_Abort)
		{
			if (m_Pause)
				continue;

			gStatus.SetPreRenderFrame();

			// CUDA time for profiling
 			CCudaTimer TmrFps;

			gSceneMutex.lock();
			SceneCopy = gScene;
			gSceneMutex.unlock();
//			SceneCopy.m_Camera.Update();

			gStatus.SetStatisticChanged("Camera", "Position", FormatVector(SceneCopy.m_Camera.m_From));
			gStatus.SetStatisticChanged("Camera", "Target", FormatVector(SceneCopy.m_Camera.m_Target));
			gStatus.SetStatisticChanged("Camera", "Up Vector", FormatVector(SceneCopy.m_Camera.m_Up));

			// Resizing the image canvas requires special attention
			if (SceneCopy.m_DirtyFlags.HasFlag(FilmResolutionDirty))
			{
				// Allocate host image buffer, this thread will blit it's frames to this buffer
				free(m_pRenderImage);
				m_pRenderImage = NULL;

				m_pRenderImage = (CColorRgbLdr*)malloc(SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorRgbLdr));

				if (m_pRenderImage)
					memset(m_pRenderImage, 0, SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorRgbLdr));
			
				gStatus.SetStatisticChanged("Host Memory", "LDR Frame Buffer", QString::number(3 * SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorRgbLdr) / MB, 'f', 2), "MB");

//				m_CudaFrameBuffers.Resize(Vec2i(SceneCopy.m_Camera.m_Film.GetWidth(), SceneCopy.m_Camera.m_Film.GetHeight()));

				BindRenderCanvasView(SceneCopy.m_Camera.m_Film.m_Resolution);

				// Reset no. iterations
				SceneCopy.SetNoIterations(0);

				Log("Render canvas resized to: " + QString::number(SceneCopy.m_Camera.m_Film.m_Resolution.GetResX()) + " x " + QString::number(SceneCopy.m_Camera.m_Film.m_Resolution.GetResY()) + " pixels", "application-resize");
			}

			// Restart the rendering when when the camera, lights and render params are dirty
			if (SceneCopy.m_DirtyFlags.HasFlag(CameraDirty | LightsDirty | RenderParamsDirty | TransferFunctionDirty))
			{
				ResetRenderCanvasView();

				// Reset no. iterations
				gScene.SetNoIterations(0);
			}

			// At this point, all dirty flags should have been taken care of, since the flags in the original scene are now cleared
			gScene.m_DirtyFlags.ClearAllFlags();

			SceneCopy.m_DenoiseParams.SetWindowRadius(4.0f);
			SceneCopy.m_DenoiseParams.m_LerpC = 0.33f * (max((float)gScene.GetNoIterations(), 1.0f) * 0.03f);//1.0f - powf(1.0f / (float)gScene.GetNoIterations(), 15.0f);//1.0f - expf(-0.01f * (float)gScene.GetNoIterations());
//			SceneCopy.m_DenoiseParams.m_Enabled = false;

			SceneCopy.m_Camera.Update();

			BindConstants(&SceneCopy);

			BindTransferFunctionOpacity(SceneCopy.m_TransferFunctions.m_Opacity);
			BindTransferFunctionDiffuse(SceneCopy.m_TransferFunctions.m_Diffuse);
			BindTransferFunctionSpecular(SceneCopy.m_TransferFunctions.m_Specular);
			BindTransferFunctionRoughness(SceneCopy.m_TransferFunctions.m_Roughness);
			BindTransferFunctionEmission(SceneCopy.m_TransferFunctions.m_Emission);

  			Render(0, SceneCopy, RenderImage, BlurImage, PostProcessImage, DenoiseImage);
		
			gScene.SetNoIterations(gScene.GetNoIterations() + 1);

			gStatus.SetStatisticChanged("Timings", "Render Image", QString::number(RenderImage.m_FilteredDuration, 'f', 2), "ms.");
			gStatus.SetStatisticChanged("Timings", "Blur Estimate", QString::number(BlurImage.m_FilteredDuration, 'f', 2), "ms.");
			gStatus.SetStatisticChanged("Timings", "Post Process Estimate", QString::number(PostProcessImage.m_FilteredDuration, 'f', 2), "ms.");
			gStatus.SetStatisticChanged("Timings", "De-noise Image", QString::number(DenoiseImage.m_FilteredDuration, 'f', 2), "ms.");

			FPS.AddDuration(1000.0f / TmrFps.ElapsedTime());

 			gStatus.SetStatisticChanged("Performance", "FPS", QString::number(FPS.m_FilteredDuration, 'f', 2), "Frames/Sec.");
 			gStatus.SetStatisticChanged("Performance", "No. Iterations", QString::number(SceneCopy.GetNoIterations()), "Iterations");

			HandleCudaError(cudaMemcpy(m_pRenderImage, GetDisplayEstimate(), SceneCopy.m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorRgbLdr), cudaMemcpyDeviceToHost));

			gFrameBuffer.Set((unsigned char*)m_pRenderImage, SceneCopy.m_Camera.m_Film.GetWidth(), SceneCopy.m_Camera.m_Film.GetHeight());

			if (m_SaveFrames.indexOf(SceneCopy.GetNoIterations()) > 0)
			{
				const QString ImageFilePath = QApplication::applicationDirPath() + "/Output/" + m_SaveBaseName + "_" + QString::number(SceneCopy.GetNoIterations()) + ".png";

				SaveImage((unsigned char*)m_pRenderImage, SceneCopy.m_Camera.m_Film.m_Resolution.GetResX(), SceneCopy.m_Camera.m_Film.m_Resolution.GetResY(), ImageFilePath);
			}

 			gStatus.SetPostRenderFrame();
		}
	}
	catch (QString* pMessage)
	{
		Log(*pMessage + ", rendering will be aborted");

		// Free render image buffer
		free(m_pRenderImage);
		m_pRenderImage = NULL;

		// Let others know that we have stopped rendering
		gStatus.SetRenderEnd();

		return;
	}

	// Free render image buffer
	free(m_pRenderImage);
	m_pRenderImage = NULL;

	UnbindDensityBuffer();

	UnbindTransferFunctionOpacity();
	UnbindTransferFunctionDiffuse();
	UnbindTransferFunctionSpecular();
	UnbindTransferFunctionRoughness();
	UnbindTransferFunctionEmission();

	// Let others know that we have stopped rendering
	gStatus.SetRenderEnd();

	Log("Device memory: " + QString::number(GetUsedCudaMemory() / MB, 'f', 2) + "/" + QString::number(GetTotalCudaMemory() / MB, 'f', 2) + " MB", "memory");

	// Clear the histogram
	gHistogram.Reset();
}

bool QRenderThread::Load(QString& FileName)
{
	m_FileName = FileName;

	// Create meta image reader
	vtkSmartPointer<vtkMetaImageReader> MetaImageReader = vtkMetaImageReader::New();

	QFileInfo FileInfo(FileName);

	if (!FileInfo.exists())
	{
		Log(QString(QFileInfo(FileName).filePath().replace("//", "/")).toAscii() + "  does not exist!", QLogger::Critical);
		return false;
	}

	Log(QString("Loading " + QFileInfo(FileName).fileName()).toAscii());

	// Exit if the reader can't read the file
	if (!MetaImageReader->CanReadFile(m_FileName.toAscii()))
	{
		Log(QString("Meta image reader can't read file " + QFileInfo(FileName).fileName()).toAscii(), QLogger::Critical);
		return false;
	}

	MetaImageReader->SetFileName(m_FileName.toAscii());

	MetaImageReader->Update();

	if (MetaImageReader->GetErrorCode() != vtkErrorCode::NoError)
	{
		Log("Error loading file " + QString(vtkErrorCode::GetStringFromErrorCode(MetaImageReader->GetErrorCode())));
		return false;
	}

	vtkSmartPointer<vtkImageCast> ImageCast = vtkImageCast::New();
	
	Log("Casting volume data type to short", "grid");

	ImageCast->SetInput(MetaImageReader->GetOutput());
	ImageCast->SetOutputScalarTypeToShort();
	ImageCast->Update();

	if (ImageCast->GetErrorCode() != vtkErrorCode::NoError)
	{
		Log("vtkImageCast error: " + QString(vtkErrorCode::GetStringFromErrorCode(MetaImageReader->GetErrorCode())));
		return false;
	}
	
	// Volume resolution
	int* pVolumeResolution = ImageCast->GetOutput()->GetExtent();
	gScene.m_Resolution.SetResXYZ(Vec3i(pVolumeResolution[1] + 1, pVolumeResolution[3] + 1, pVolumeResolution[5] + 1));

	Log("Resolution: " + FormatSize(gScene.m_Resolution.GetResXYZ()) + "", "grid");

	// Intensity range
	double* pIntensityRange = ImageCast->GetOutput()->GetScalarRange();
	gScene.m_IntensityRange.SetMin((float)pIntensityRange[0]);
	gScene.m_IntensityRange.SetMax((float)pIntensityRange[1]);

	Log("Intensity range: [" + QString::number(gScene.m_IntensityRange.GetMin()) + ", " + QString::number(gScene.m_IntensityRange.GetMax()) + "]", "grid");

	// Spacing
	double* pSpacing = ImageCast->GetOutput()->GetSpacing();

	gScene.m_Spacing.x = (float)pSpacing[0];
	gScene.m_Spacing.y = (float)pSpacing[1];
	gScene.m_Spacing.z = (float)pSpacing[2];

	Log("Spacing: " + FormatSize(gScene.m_Spacing, 2), "grid");

	// Compute physical size
	const Vec3f PhysicalSize(Vec3f(gScene.m_Spacing.x * (float)gScene.m_Resolution.GetResX(), gScene.m_Spacing.y * (float)gScene.m_Resolution.GetResY(), gScene.m_Spacing.z * (float)gScene.m_Resolution.GetResZ()));

	// Compute the volume's bounding box
	gScene.m_BoundingBox.m_MinP	= Vec3f(0.0f);
	gScene.m_BoundingBox.m_MaxP	= PhysicalSize / PhysicalSize.Max();

	gScene.m_GradientDelta = 1.0f / (float)gScene.m_Resolution.GetMax();
	
	Log("Bounding box: " + FormatVector(gScene.m_BoundingBox.m_MinP, 2) + " - " + FormatVector(gScene.m_BoundingBox.m_MaxP), "grid");
	
	const int DensityBufferSize = gScene.m_Resolution.GetNoElements() * sizeof(short);

 	m_pDensityBuffer = (short*)malloc(DensityBufferSize);
  	memcpy(m_pDensityBuffer, ImageCast->GetOutput()->GetScalarPointer(), DensityBufferSize);

	// Gradient magnitude volume
	vtkSmartPointer<vtkImageGradientMagnitude> GradientMagnitude = vtkImageGradientMagnitude::New();
	
	Log("Creating gradient magnitude volume", "grid");
		
	GradientMagnitude->SetDimensionality(3);
	GradientMagnitude->SetInput(ImageCast->GetOutput());
	GradientMagnitude->Update();

	vtkImageData* GradientMagnitudeBuffer = GradientMagnitude->GetOutput();
	
	// Scalar range of the gradient magnitude
	double* pGradientMagnitudeRange = GradientMagnitudeBuffer->GetScalarRange();
	
	gScene.m_GradientMagnitudeRange.SetMin((float)pGradientMagnitudeRange[0]);
	gScene.m_GradientMagnitudeRange.SetMax((float)pGradientMagnitudeRange[1]);
	
	Log("Gradient magnitude range: [" + QString::number(gScene.m_GradientMagnitudeRange.GetMin(), 'f', 2) + " - " + QString::number(gScene.m_GradientMagnitudeRange.GetMax(), 'f', 2) + "]", "grid");
	
	const int GradientMagnitudeBufferSize = gScene.m_Resolution.GetNoElements() * sizeof(short);
	
	m_pGradientMagnitudeBuffer = (short*)malloc(GradientMagnitudeBufferSize);
	memcpy(m_pGradientMagnitudeBuffer, GradientMagnitudeBuffer->GetScalarPointer(), GradientMagnitudeBufferSize);

	// Build the histogram
	Log("Creating gradient magnitude histogram", "grid");

	vtkSmartPointer<vtkImageAccumulate> GradMagHistogram = vtkSmartPointer<vtkImageAccumulate>::New();

	GradMagHistogram->SetInputConnection(GradientMagnitude->GetOutputPort());
	GradMagHistogram->SetComponentExtent(0, 255, 0, 0, 0, 0);
	GradMagHistogram->SetComponentOrigin(0, 0, 0);
	GradMagHistogram->SetComponentSpacing(gScene.m_GradientMagnitudeRange.GetRange() / 256.0f, 0, 0);
//	GradMagHistogram->IgnoreZeroOn();
	GradMagHistogram->Update();

	gScene.m_GradMagMean = (float)GradMagHistogram->GetMean()[0];
	gScene.m_GradientFactor = gScene.m_GradMagMean;

	Log("Mean gradient magnitude: " + QString::number(gScene.m_GradMagMean, 'f', 2), "grid");

	Log("Creating density histogram", "grid");

	// Build the histogram
	vtkSmartPointer<vtkImageAccumulate> Histogram = vtkSmartPointer<vtkImageAccumulate>::New();

	Log("Creating histogram", "grid");

 	Histogram->SetInputConnection(ImageCast->GetOutputPort());
 	Histogram->SetComponentExtent(0, 256, 0, 0, 0, 0);
 	Histogram->SetComponentOrigin(gScene.m_IntensityRange.GetMin(), 0, 0);
 	Histogram->SetComponentSpacing(gScene.m_IntensityRange.GetRange() / 256.0f, 0, 0);
 	Histogram->IgnoreZeroOn();
 	Histogram->Update();
 
	// Update the histogram in the transfer function
	gHistogram.SetBins((int*)Histogram->GetOutput()->GetScalarPointer(), 256);
	
	gStatus.SetStatisticChanged("Volume", "File", QFileInfo(m_FileName).fileName(), "");
	gStatus.SetStatisticChanged("Volume", "Bounding Box", "", "");
	gStatus.SetStatisticChanged("Bounding Box", "Min", FormatVector(gScene.m_BoundingBox.m_MinP, 2), "m");
	gStatus.SetStatisticChanged("Bounding Box", "Max", FormatVector(gScene.m_BoundingBox.m_MaxP, 2), "m");
	gStatus.SetStatisticChanged("Volume", "Physical Size", FormatSize(PhysicalSize, 2), "mm");
	gStatus.SetStatisticChanged("Volume", "Resolution", FormatSize(gScene.m_Resolution.GetResXYZ()), "Voxels");
	gStatus.SetStatisticChanged("Volume", "Spacing", FormatSize(gScene.m_Spacing, 2), "mm");
	gStatus.SetStatisticChanged("Volume", "No. Voxels", QString::number(gScene.m_Resolution.GetNoElements()), "Voxels");
	gStatus.SetStatisticChanged("Volume", "Density Range", "[" + QString::number(gScene.m_IntensityRange.GetMin()) + ", " + QString::number(gScene.m_IntensityRange.GetMax()) + "]", "");
	
	return true;
}

void QRenderThread::OnUpdateTransferFunction(void)
{
	QMutexLocker Locker(&gSceneMutex);

	QTransferFunction TransferFunction = gTransferFunction;

	gScene.m_TransferFunctions.m_Opacity.m_NoNodes		= TransferFunction.GetNodes().size();
	gScene.m_TransferFunctions.m_Diffuse.m_NoNodes		= TransferFunction.GetNodes().size();
	gScene.m_TransferFunctions.m_Specular.m_NoNodes		= TransferFunction.GetNodes().size();
	gScene.m_TransferFunctions.m_Emission.m_NoNodes		= TransferFunction.GetNodes().size();
	gScene.m_TransferFunctions.m_Roughness.m_NoNodes	= TransferFunction.GetNodes().size();

	for (int i = 0; i < TransferFunction.GetNodes().size(); i++)
	{
		QNode& Node = TransferFunction.GetNode(i);

		const float Intensity = Node.GetIntensity();

		// Positions
		gScene.m_TransferFunctions.m_Opacity.m_P[i]		= Intensity;
		gScene.m_TransferFunctions.m_Diffuse.m_P[i]		= Intensity;
		gScene.m_TransferFunctions.m_Specular.m_P[i]	= Intensity;
		gScene.m_TransferFunctions.m_Emission.m_P[i]	= Intensity;
		gScene.m_TransferFunctions.m_Roughness.m_P[i]	= Intensity;

		// Colors
		gScene.m_TransferFunctions.m_Opacity.m_C[i]		= CColorRgbHdr(Node.GetOpacity());
		gScene.m_TransferFunctions.m_Diffuse.m_C[i]		= CColorRgbHdr(Node.GetDiffuse().redF(), Node.GetDiffuse().greenF(), Node.GetDiffuse().blueF());
		gScene.m_TransferFunctions.m_Specular.m_C[i]	= CColorRgbHdr(Node.GetSpecular().redF(), Node.GetSpecular().greenF(), Node.GetSpecular().blueF());
		gScene.m_TransferFunctions.m_Emission.m_C[i]	= 500.0f * CColorRgbHdr(Node.GetEmission().redF(), Node.GetEmission().greenF(), Node.GetEmission().blueF());

		const float Roughness = 1.0f - expf(-Node.GetGlossiness());

		gScene.m_TransferFunctions.m_Roughness.m_C[i]	= CColorRgbHdr(Roughness * 100);
	}

	gScene.m_DensityScale	= TransferFunction.GetDensityScale();
	gScene.m_ShadingType	= TransferFunction.GetShadingType();
	gScene.m_GradientFactor	= TransferFunction.GetGradientFactor();

	gScene.m_DirtyFlags.SetFlag(TransferFunctionDirty);

	/*
	FILE * pFile;
	int n;
	char name [100];

	pFile = fopen ("c:\\tf.txt","w");

	if (pFile)
	{
		for (int i = 0; i < 255; i++)
		{
			fprintf(pFile, "%0.2f\n", gScene.m_TransferFunctions.m_Roughness.F((float)i / 255.0f));
		}
	}

	fclose (pFile);
	*/
}

void QRenderThread::OnUpdateCamera(void)
{
	QMutexLocker Locker(&gSceneMutex);

	gScene.m_Camera.m_Film.m_Exposure = 1.0f - gCamera.GetFilm().GetExposure();

	if (gCamera.GetFilm().IsDirty())
	{
		const int FilmWidth	= gCamera.GetFilm().GetWidth();
		const int FilmHeight = gCamera.GetFilm().GetHeight();

 		gScene.m_Camera.m_Film.m_Resolution.SetResX(FilmWidth);
		gScene.m_Camera.m_Film.m_Resolution.SetResY(FilmHeight);
		gScene.m_Camera.Update();
		gCamera.GetFilm().UnDirty();
// 		// 
 		gScene.m_DirtyFlags.SetFlag(FilmResolutionDirty);
	}

// 	gScene.m_Camera.m_From	= gCamera.GetFrom();
// 	gScene.m_Camera.m_Target	= gCamera.GetTarget();
// 	gScene.m_Camera.m_Up		= gCamera.GetUp();

	gScene.m_Camera.Update();

	// Aperture
	gScene.m_Camera.m_Aperture.m_Size	= gCamera.GetAperture().GetSize();

	// Projection
	gScene.m_Camera.m_FovV = gCamera.GetProjection().GetFieldOfView();

	// Focus
	gScene.m_Camera.m_Focus.m_Type			= (CFocus::EType)gCamera.GetFocus().GetType();
	gScene.m_Camera.m_Focus.m_FocalDistance = gCamera.GetFocus().GetFocalDistance();

	gScene.m_DenoiseParams.m_Enabled = gCamera.GetFilm().GetNoiseReduction();

	gScene.m_DirtyFlags.SetFlag(CameraDirty);
}

void QRenderThread::OnUpdateLighting(void)
{
	QMutexLocker Locker(&gSceneMutex);

	gScene.m_Lighting.Reset();

	if (gLighting.Background().GetEnabled())
	{
		CLight BackgroundLight;

		BackgroundLight.m_T	= 1;

		BackgroundLight.m_ColorTop		= gLighting.Background().GetIntensity() * CColorRgbHdr(gLighting.Background().GetTopColor().redF(), gLighting.Background().GetTopColor().greenF(), gLighting.Background().GetTopColor().blueF());
		BackgroundLight.m_ColorMiddle	= gLighting.Background().GetIntensity() * CColorRgbHdr(gLighting.Background().GetMiddleColor().redF(), gLighting.Background().GetMiddleColor().greenF(), gLighting.Background().GetMiddleColor().blueF());
		BackgroundLight.m_ColorBottom	= gLighting.Background().GetIntensity() * CColorRgbHdr(gLighting.Background().GetBottomColor().redF(), gLighting.Background().GetBottomColor().greenF(), gLighting.Background().GetBottomColor().blueF());

		BackgroundLight.Update(gScene.m_BoundingBox);

		gScene.m_Lighting.AddLight(BackgroundLight);
	}

	for (int i = 0; i < gLighting.GetLights().size(); i++)
	{
		QLight& Light = gLighting.GetLights()[i];

		CLight AreaLight;

		AreaLight.m_T			= 0;
		AreaLight.m_Theta		= Light.GetTheta() / RAD_F;
		AreaLight.m_Phi			= Light.GetPhi() / RAD_F;
		AreaLight.m_Width		= Light.GetWidth();
		AreaLight.m_Height		= Light.GetHeight();
		AreaLight.m_Distance	= Light.GetDistance();
		AreaLight.m_Color		= Light.GetIntensity() * CColorRgbHdr(Light.GetColor().redF(), Light.GetColor().greenF(), Light.GetColor().blueF());

		AreaLight.Update(gScene.m_BoundingBox);

		gScene.m_Lighting.AddLight(AreaLight);
	}

	gScene.m_DirtyFlags.SetFlag(LightsDirty);
}

void QRenderThread::OnRenderPause(const bool& Pause)
{
	m_Pause = Pause;
}

CColorRgbLdr* QRenderThread::GetRenderImage(void) const
{
	return m_pRenderImage;
}

void StartRenderThread(QString& FileName)
{
	// Create new render thread
 	gpRenderThread = new QRenderThread(FileName);
  //	gpRenderThread->setStackSize(1500000000);

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



#include "LoadSettingsDialog.h"
#include "MainWindow.h"
#include "LoadVolume.h"
#include "RenderThread.h"
#include "Scene.h"
#include "TransferFunction.h"

// VTK includes
#include <vtkSmartPointer.h>
#include <vtkMetaImageReader.h>
#include <vtkImageCast.h>
#include <vtkImageResample.h>
#include <vtkImageData.h>
#include <vtkImageGradientMagnitude.h>
#include <vtkCallbackCommand.h>
#include <vtkImageAccumulate.h>
#include <vtkIntArray.h>

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

bool LoadVtkVolume(const char* pFile, CScene* pScene, vtkImageData*& pImageDataVolume)
{
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
	if (!MetaImageReader->CanReadFile(pFile))
		return false;

	// Create progress callback
	vtkSmartPointer<vtkCallbackCommand> ProgressCallback = vtkSmartPointer<vtkCallbackCommand>::New();

	// Set callback
	ProgressCallback->SetCallback (OnProgress);
	ProgressCallback->SetClientData(MetaImageReader);

	// Progress handling
//	MetaImageReader->AddObserver(vtkCommand::ProgressEvent, ProgressCallback);

	MetaImageReader->SetFileName(pFile);

	MetaImageReader->Update();

	vtkSmartPointer<vtkImageCast> pImageCast = vtkImageCast::New();

	pImageCast->SetOutputScalarTypeToShort();
	pImageCast->SetInput(MetaImageReader->GetOutput());
	
	pImageCast->Update();

	pImageDataVolume = pImageCast->GetOutput();

	/*
//	if (LoadSettingsDialog.GetResample())
//	{
		// Create resampler
		vtkSmartPointer<vtkImageResample> ImageResample = vtkImageResample::New();

		// Progress handling
		ImageResample->AddObserver(vtkCommand::ProgressEvent, ProgressCallback);

		ImageResample->SetInput(pImageDataVolume);

		// Obtain resampling scales from dialog input
		gpScene->m_Scale.x = LoadSettingsDialog.GetResampleX();
		gpScene->m_Scale.y = LoadSettingsDialog.GetResampleY();
		gpScene->m_Scale.z = LoadSettingsDialog.GetResampleZ();

		// Apply scaling factors
		ImageResample->SetAxisMagnificationFactor(0, gpScene->m_Scale.x);
		ImageResample->SetAxisMagnificationFactor(1, gpScene->m_Scale.y);
		ImageResample->SetAxisMagnificationFactor(2, gpScene->m_Scale.z);
	
		// Resample
		ImageResample->Update();

		pImageDataVolume = ImageResample->GetOutput();
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

	

	pScene->m_MemorySize	= (float)pImageDataVolume->GetActualMemorySize() / 1024.0f;
	

	double Range[2];

	pImageDataVolume->GetScalarRange(Range);

	pScene->m_IntensityRange.m_Min	= (float)Range[0];
	pScene->m_IntensityRange.m_Max	= (float)Range[1];

	gTransferFunction.SetRangeMin((float)Range[0]);
	gTransferFunction.SetRangeMax((float)Range[1]);

	int* pExtent = pImageDataVolume->GetExtent();
	
	pScene->m_Resolution.m_XYZ.x = pExtent[1] + 1;
	pScene->m_Resolution.m_XYZ.y = pExtent[3] + 1;
	pScene->m_Resolution.m_XYZ.z = pExtent[5] + 1;
	pScene->m_Resolution.Update();

	double* pSpacing = pImageDataVolume->GetSpacing();

	
	pScene->m_Spacing.x = pSpacing[0];
	pScene->m_Spacing.y = pSpacing[1];
	pScene->m_Spacing.z = pSpacing[2];
	

	Vec3f Resolution = Vec3f(pScene->m_Spacing.x * (float)pScene->m_Resolution.m_XYZ.x, pScene->m_Spacing.y * (float)pScene->m_Resolution.m_XYZ.y, pScene->m_Spacing.z * (float)pScene->m_Resolution.m_XYZ.z);

	float Max = Resolution.Max();

	pScene->m_NoVoxels				= pScene->m_Resolution.m_NoElements;
	pScene->m_BoundingBox.m_MinP	= Vec3f(0.0f);
	pScene->m_BoundingBox.m_MaxP	= Vec3f(Resolution.x / Max, Resolution.y / Max, Resolution.z / Max);

	// Build the histogram
	vtkSmartPointer<vtkImageAccumulate> Histogram = vtkSmartPointer<vtkImageAccumulate>::New();
	Histogram->SetInputConnection(ImageResample->GetOutputPort());
	Histogram->SetComponentExtent(0, 1024, 0, 0, 0, 0);
	Histogram->SetComponentOrigin(0, 0, 0);
	Histogram->SetComponentSpacing(1, 0, 0);
	Histogram->IgnoreZeroOn();
	Histogram->Update();
 
	// Update the histogram in the transfer function
//	gTransferFunction.SetHistogram((int*)Histogram->GetOutput()->GetScalarPointer(), 256);
	
	// Delete progress dialog
//	gpProgressDialog->close();
//	delete gpProgressDialog;
//	gpProgressDialog = NULL;

	return true;
}
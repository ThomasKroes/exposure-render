#pragma once

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

// http://www.na-mic.org/svn/Slicer3/branches/cuda/Modules/VolumeRenderingCuda/

/*
class EXPOSURE_RENDER_DLL vtkVolumeCudaMapper : public vtkVolumeMapper
{
public:
    vtkTypeRevisionMacro(vtkVolumeCudaMapper,vtkVolumeMapper);
    static vtkVolumeCudaMapper *New();

    virtual void SetInput( vtkImageData * );
    virtual void Render(vtkRenderer *, vtkVolume *);

    // Should be in Property??
    void SetThreshold(unsigned int min, unsigned int max);
    void SetThreshold(double* range) { SetThreshold((unsigned int)range[0], (unsigned int)range[1]); }

   //BTX
   void SetRenderMode(int mode);
   int GetCurrentRenderMode() const;// { return this->CurrentRenderMode; }
   //ETX

   vtkImageData* GetOutput() { return NULL;  }

   void PrintSelf(ostream& os, vtkIndent indent);

protected:
    vtkVolumeCudaMapper();
    virtual ~vtkVolumeCudaMapper();

    void UpdateOutputResolution(unsigned int width, unsigned int height, bool TypeChanged = false);

//    vtkCudaRendererInformationHandler* RendererInfoHandler;
//    vtkCudaVolumeInformationHandler* VolumeInfoHandler;

private:
    vtkVolumeCudaMapper operator=(const vtkVolumeCudaMapper&);
    vtkVolumeCudaMapper(const vtkVolumeCudaMapper&);
};
*/
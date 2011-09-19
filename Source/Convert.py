from vtk import*

import os

Path = "C:\\Volumes"  # insert the path to the directory of interest
DirList = os.listdir(Path)
  
print "Removing pre-existing resampled volumes"

for FileName in DirList:
    BaseName, Extension = os.path.splitext(FileName)
    if (BaseName.endswith("_small")):
        print "Removing: " + BaseName + Extension
        os.remove(Path + "\\" + FileName)

print "Resampling volumes"

for FileName in DirList:
    BaseName, Extension = os.path.splitext(FileName)
    if (Extension == ".mhd" and not BaseName.endswith("_small")):

        print("Reading meta image " + FileName)
        
        Reader = vtkMetaImageReader()

        MhdFileName = Path + "\\" + FileName
        
        Reader.SetFileName(MhdFileName)
        Reader.Update()

        Resample = vtkImageResample()

        ResampleFactor = 0.5

        print("Resampling image at " + str(100.0 * ResampleFactor) + "% of original volume")
        
        Resample.SetInput(Reader.GetOutput())
        Resample.SetAxisMagnificationFactor(0, ResampleFactor)
        Resample.SetAxisMagnificationFactor(1, ResampleFactor)
        Resample.SetAxisMagnificationFactor(2, ResampleFactor)

        Resample.Update()

        MetaImageWriter = vtkMetaImageWriter()

        MetaImageWriter.SetFileName(Path + "\\" + BaseName + "_small.mhd")
        MetaImageWriter.SetRAWFileName(Path + "\\" + BaseName + "_small.raw")
        
        MetaImageWriter.SetInput(Resample.GetOutput())

        print("Writing resampled " + Path + "\\" + BaseName + "_small.mhd")
        print("Writing resampled " + Path + "\\" + BaseName + "_small.raw")
        
        MetaImageWriter.Write()

print "Done with re-sampling"



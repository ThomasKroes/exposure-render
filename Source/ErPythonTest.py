#!/usr/bin/env python


#import vtkErCorePython
import vtkmyImagingPython #ErCorePython
import vtk.rendering
# Create the standard renderer, render window and interactor
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

ren.SetBackground(1, 1, 1)
renWin.SetSize(600, 600)
renWin.Render()

#ErCamera = vtkErCorePython.vtkErVolumeMapper()

# vtkRenderingPythonD vtkGraphicsPythonD vtkIOPythonD vtkWidgetsPythonD vtkCommonPythonD vtkVolumeRenderingPythonD vtkViewsPythonD

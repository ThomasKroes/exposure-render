#pragma once

#include "Geometry.h"

// Qt
#include <QtGui>

class vtkImageData;
class CScene;

bool LoadVtkVolume(const char* pFile, CScene* pScene, vtkImageData*& pImageDataVolume);
/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Stable.h"

#include "vtkCudaMemoryTexture.h"

#include <vtkObjectFactory.h>
#include <vtkgl.h>
#include <vtkOpenGLExtensionManager.h>

#include "cuda_runtime_api.h"
#include "cuda_gl_interop.h"

vtkCxxRevisionMacro(vtkCudaMemoryTexture, "$Revision 1.0 $");
vtkStandardNewMacro(vtkCudaMemoryTexture);

bool vtkCudaMemoryTexture::GLBufferObjectsAvailiable = false;

vtkCudaMemoryTexture::vtkCudaMemoryTexture()
{
    Initialize();
}

vtkCudaMemoryTexture::~vtkCudaMemoryTexture()
{
    if (TextureID == 0 || !glIsTexture(TextureID))
        glGenTextures(1, &TextureID);

    if (vtkCudaMemoryTexture::GLBufferObjectsAvailiable == true)
        if (BufferObjectID != 0 && vtkgl::IsBufferARB(BufferObjectID))
            vtkgl::DeleteBuffersARB(1, &BufferObjectID);
}

void vtkCudaMemoryTexture::Initialize()
{
    TextureID		= 0;
    BufferObjectID	= 0;
    Height = Width	= 0;
    RenderDestination = NULL;

    if (vtkCudaMemoryTexture::GLBufferObjectsAvailiable == false)
    {
        // check for the RenderMode
        vtkOpenGLExtensionManager *extensions = vtkOpenGLExtensionManager::New();
        extensions->SetRenderWindow(NULL);
        if (extensions->ExtensionSupported("GL_ARB_vertex_buffer_object"))
        {
            extensions->LoadExtension("GL_ARB_vertex_buffer_object");
            vtkCudaMemoryTexture::GLBufferObjectsAvailiable = true;
            CurrentRenderMode = RenderToTexture;
        }
        else
        {
            CurrentRenderMode = RenderToMemory;
        }
        extensions->Delete();
    }
}

void vtkCudaMemoryTexture::SetSize(unsigned int width, unsigned int height)
{
//	m_FrameBuffer.Resize(CResolution2D(width, height));
	m_Host.Resize(CResolution2D(width, height));

	RebuildBuffer();
}

void vtkCudaMemoryTexture::RebuildBuffer()
{
    glEnable(GL_TEXTURE_2D);
    
	if (TextureID != 0 && glIsTexture(TextureID))
        glDeleteTextures(1, &TextureID);

    glGenTextures(1, &TextureID);
    glBindTexture(GL_TEXTURE_2D, TextureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, Width, Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, m_Host.GetPtr(0));
    glBindTexture(GL_TEXTURE_2D, 0);

    if (CurrentRenderMode == RenderToTexture)
    {
        // OpenGL Buffer Code
        if (BufferObjectID != 0 && vtkgl::IsBufferARB(BufferObjectID))
            vtkgl::DeleteBuffersARB(1, &BufferObjectID);

        vtkgl::GenBuffersARB(1, &BufferObjectID);
        vtkgl::BindBufferARB(vtkgl::PIXEL_UNPACK_BUFFER_ARB, BufferObjectID);
        vtkgl::BufferDataARB(vtkgl::PIXEL_UNPACK_BUFFER_ARB, Width * Height * 4, m_Host.GetPtr(0), vtkgl::STREAM_COPY);
        vtkgl::BindBufferARB(vtkgl::PIXEL_UNPACK_BUFFER_ARB, 0);
    }
}
void vtkCudaMemoryTexture::SetRenderMode(int mode)
{
    if (mode == RenderToTexture && vtkCudaMemoryTexture::GLBufferObjectsAvailiable)
    {
        CurrentRenderMode = mode;
    }
    else
    {
        CurrentRenderMode = RenderToMemory;
    }

    RebuildBuffer();
}

void vtkCudaMemoryTexture::BindTexture()
{
    glPushAttrib(GL_ENABLE_BIT);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, TextureID);
}

void vtkCudaMemoryTexture::UnbindTexture()
{
    glPopAttrib();
}

void vtkCudaMemoryTexture::BindBuffer()
{
    if (CurrentRenderMode == RenderToTexture)
    {
//		cudaGLSetGLDevice(0);
//		cudaGLRegisterBufferObject(BufferObjectID);

//        vtkgl::BindBufferARB(vtkgl::PIXEL_UNPACK_BUFFER_ARB, BufferObjectID);
//        HandleCudaError(cudaGLMapBufferObject((void**)&RenderDestination, BufferObjectID));
		/**/
    }
    else
    {
//        RenderDestination = CudaOutputData.GetMemPointerAs<unsigned char>();
    }
}

void vtkCudaMemoryTexture::UnbindBuffer()
{
    if (CurrentRenderMode == RenderToTexture)
    {
		/*
        HandleCudaError(cudaGLUnmapBufferObject(BufferObjectID));
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, Width, Height, GL_RGBA, GL_UNSIGNED_BYTE, (0));
        HandleCudaError( cudaGLUnregisterBufferObject(BufferObjectID) );
        vtkgl::BindBufferARB(vtkgl::PIXEL_UNPACK_BUFFER_ARB, 0);
		*/
    }
    else // (CurrentRenderMode == RenderToMemory)
    {
//        CudaOutputData.CopyTo(&LocalOutputData);
//        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, Width, Height, GL_RGBA, GL_UNSIGNED_BYTE, LocalOutputData.GetMemPointer());
    }
    RenderDestination = NULL;
}

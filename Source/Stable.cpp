/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Stable.h"

QString GetOpenFileName(const QString& Caption, const QString& Filter, const QString& Icon)
{
	QFileDialog FileDialog;

	FileDialog.setWindowTitle(Caption);
	FileDialog.setFilter(Filter);
	FileDialog.setOption(QFileDialog::DontUseNativeDialog, true);
	FileDialog.setWindowIcon(Icon.isEmpty() ? GetIcon("disk") : GetIcon(Icon));

	if (FileDialog.exec() == QMessageBox::Rejected)
		return "";

	return FileDialog.selectedFiles().value(0);
}

QString GetSaveFileName(const QString& Caption, const QString& Filter, const QString& Icon)
{
	QFileDialog FileDialog;

	FileDialog.setWindowTitle(Caption);
	FileDialog.setFilter(Filter);
	FileDialog.setOption(QFileDialog::DontUseNativeDialog, true);
	FileDialog.setWindowIcon(Icon.isEmpty() ? GetIcon("disk") : GetIcon(Icon));
	FileDialog.setAcceptMode(QFileDialog::AcceptSave);

	if (FileDialog.exec() == QMessageBox::Rejected)
		return "";

	return FileDialog.selectedFiles().value(0);
}

void SaveImage(const unsigned char* pImageBuffer, const int& Width, const int& Height, QString FilePath /*= ""*/)
{
	if (!pImageBuffer)
	{
		Log("Can't save image, buffer is empty", QLogger::Critical);
		return;
	}

	if (FilePath.isEmpty())
		FilePath = GetSaveFileName("Save Image", "PNG Files (*.png)", "image-export");

	if (!FilePath.isEmpty())
	{
		QImage* pTempImage = new QImage(pImageBuffer, Width, Height,  QImage::Format_RGB888);

		if (!pTempImage->save(FilePath, "PNG") )
			Log("Unable to save image");
		else
			Log(FilePath + " saved", "image-export");

		delete pTempImage;
	}
	else
	{
		Log("Can't save image, file path is empty", QLogger::Critical);
	}
}

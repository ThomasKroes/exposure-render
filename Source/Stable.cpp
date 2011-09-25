
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

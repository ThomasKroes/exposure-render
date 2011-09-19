
#include "Stable.h"

QString GetOpenFileName(const QString& Caption, const QString& Filter)
{
	QFileDialog FileDialog;

	FileDialog.setWindowTitle(Caption);
	FileDialog.setFilter(Filter);
	FileDialog.setOption(QFileDialog::DontUseNativeDialog, true);
	FileDialog.setWindowIcon(GetIcon("folder-open-document"));

	if (FileDialog.exec() == QMessageBox::Rejected)
		return "";

	return FileDialog.selectedFiles().value(0);
}

QString GetSaveFileName(const QString& Caption, const QString& Filter)
{
	QFileDialog FileDialog;

	FileDialog.setWindowTitle(Caption);
	FileDialog.setFilter(Filter);
	FileDialog.setOption(QFileDialog::DontUseNativeDialog, true);
	FileDialog.setWindowIcon(GetIcon("disk"));

	if (FileDialog.exec() == QMessageBox::Rejected)
		return "";

	return FileDialog.selectedFiles().value(0);
}

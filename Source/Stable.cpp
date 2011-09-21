
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


#include <QtGui>

#include "MainWindow.h"
#include "TransferFunction.h"

#include "cutil_inline.h"

int main(int ArgumentCount, char* pArgv[])
{
	// Create the application
    QApplication Application(ArgumentCount, pArgv);

	// Adjust style
	Application.setStyle("plastique");
	Application.setOrganizationName("TU Delft");
	Application.setApplicationName("Exposure");

	CMainWindow MainWindow;

	// Show the main window
	gpMainWindow = &MainWindow;
    MainWindow.show();

	int Result = Application.exec();

	return Result;
}

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

	// Show the main window
	gpMainWindow = new CMainWindow();
    gpMainWindow->show();

	int Result = Application.exec();

	delete gpMainWindow;

	// Remove render thread
	if (gpRenderThread)
	{
		gThreadAlive = false;
		gpRenderThread->terminate();
		while(!gpRenderThread->isFinished()){}
		delete gpRenderThread;
		gpRenderThread = NULL;
	}

    return Result;
}
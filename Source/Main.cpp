
#include <QtGui>

#include "MainWindow.h"
#include "TransferFunction.h"

#include "cutil_inline.h"

int main(int ArgumentCount, char* pArgv[])
{
	// Create the application
    QApplication Application(ArgumentCount, pArgv);

	QSplashScreen SplashScreen;

	SplashScreen.show();

	Qt::Alignment TopRight = Qt::AlignRight | Qt::AlignTop;

	SplashScreen.showMessage("Setting up main window", TopRight, Qt::white);
	
	Q_INIT_RESOURCE(icons);

	// Adjust style
	Application.setStyle("plastique");
	Application.setOrganizationName("TU Delft");
	Application.setApplicationName("Exposure");

	CMainWindow MainWindow;

	SplashScreen.showMessage("Establishing connections", TopRight, Qt::white);

	// Show the main window
	gpMainWindow = &MainWindow;
    MainWindow.show();

	SplashScreen.finish(&MainWindow);

	int Result = Application.exec();

	return Result;
}
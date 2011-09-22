
// Precompiled headers
#include "Stable.h"

#include "MainWindow.h"

int main(int ArgumentCount, char* pArgv[])
{
	// Create the application
    QApplication Application(ArgumentCount, pArgv);

	// Adjust style
	Application.setStyle("plastique");
	Application.setOrganizationName("TU Delft");
	Application.setApplicationName("Exposure Render");

	// Application settings
	QSettings Settings;

	Settings.setValue("version", "1.0.0");

	// Main window
	CMainWindow MainWindow;

	// Show the main window
	gpMainWindow = &MainWindow;

	// Show it
	MainWindow.show();

	// Override the application setting to enforce the display of the startup dialog
	Settings.setValue("startup/dialog/show", QVariant(true));

	// Show startup dialog
	if (Settings.value("startup/dialog/show").toBool() == true)
		MainWindow.ShowStartupDialog();

	// Execute the application
	int Result = Application.exec();

	return Result;
}
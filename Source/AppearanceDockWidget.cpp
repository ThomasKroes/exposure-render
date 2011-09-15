
// Precompiled headers
#include "Stable.h"

#include "AppearanceDockWidget.h"
#include "MainWindow.h"

QAppearanceWidget::QAppearanceWidget(QWidget* pParent) :
	QWidget(pParent),
	m_MainLayout(),
	m_TransferFunctionWidget(),
	m_NodePropertiesWidget(),
	m_PresetsWidget(NULL, "Appearance", "Appearance")
{
	// Create main layout
	m_MainLayout.setAlignment(Qt::AlignTop);
	setLayout(&m_MainLayout);

	m_MainLayout.addWidget(&m_TransferFunctionWidget, 1, 0);
	m_MainLayout.addWidget(&m_NodePropertiesWidget, 2, 0);
	m_MainLayout.addWidget(&m_PresetsWidget, 0, 0);

	// Inform us when a new preset is loaded, when we need to save a preset and when the rendering begins and ends
	connect(&m_PresetsWidget, SIGNAL(LoadPreset(const QString&)), this, SLOT(OnLoadPreset(const QString&)));
	connect(&m_PresetsWidget, SIGNAL(SavePreset(const QString&)), this, SLOT(OnSavePreset(const QString&)));
	connect(&gRenderStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
	connect(&gRenderStatus, SIGNAL(RenderEnd()), this, SLOT(OnRenderEnd()));
}

void QAppearanceWidget::OnLoadPreset(const QString& Name)
{
	gTransferFunction = m_PresetsWidget.GetPreset(Name);

	// De-normalize node intensity
	gTransferFunction.DeNormalizeIntensity();
}

void QAppearanceWidget::OnSavePreset(const QString& Name)
{
	QTransferFunction Preset = gTransferFunction;
	Preset.SetName(Name);

	// Normalize node intensity
	Preset.NormalizeIntensity();

	// Save the preset
	m_PresetsWidget.SavePreset(Preset);
}

void QAppearanceWidget::OnRenderBegin(void)
{
	// Add a default transfer function and load it
	m_PresetsWidget.InsertPreset(0, QTransferFunction::Default());
	m_PresetsWidget.LoadPreset("Default");
}

void QAppearanceWidget::OnRenderEnd(void)
{
	// Add a default transfer function and load it
	m_PresetsWidget.InsertPreset(0, QTransferFunction::Default());
	m_PresetsWidget.LoadPreset("Default");
}

QAppearanceDockWidget::QAppearanceDockWidget(QWidget *parent) :
	QDockWidget(parent),
	m_VolumeAppearanceWidget()
{
	setWindowTitle("Appearance");
	setToolTip("<img src=':/Images/palette.png'><div>Volume Appearance</div>");
	setWindowIcon(GetIcon("palette"));

	m_VolumeAppearanceWidget.setParent(this);

	setWidget(&m_VolumeAppearanceWidget);
}
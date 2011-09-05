
#include "FocusWidget.h"
#include "MainWindow.h"
#include "RenderThread.h"

CFocusWidget::CFocusWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_GridLayout(),
	m_FocusTypeComboBox(),
	m_FocalDistanceSlider(),
	m_FocalDistanceSpinBox()
{
	setTitle("Focus");
	setStatusTip("Focus properties");
	setToolTip("Focus properties");

	setLayout(&m_GridLayout);

	// Focus type
	m_GridLayout.addWidget(new QLabel("Focus type"), 5, 0);

	m_FocusTypeComboBox.addItem("Automatic");
	m_FocusTypeComboBox.addItem("Pick");
	m_FocusTypeComboBox.addItem("Manual");
	m_GridLayout.addWidget(&m_FocusTypeComboBox, 5, 1, 1, 2);
	
	connect(&m_FocusTypeComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(SetFocusType(int)));

	// Focal distance
	m_GridLayout.addWidget(new QLabel("Focal distance"), 6, 0);

	m_FocalDistanceSlider.setOrientation(Qt::Orientation::Horizontal);
	m_FocalDistanceSlider.setEnabled(false);
    m_FocalDistanceSlider.setFocusPolicy(Qt::StrongFocus);
    m_FocalDistanceSlider.setTickPosition(QDoubleSlider::TickPosition::NoTicks);
	m_FocalDistanceSlider.setRange(0.0, 1000000.0);
	m_GridLayout.addWidget(&m_FocalDistanceSlider, 6, 1);
	
	m_FocalDistanceSpinBox.setEnabled(false);
    m_FocalDistanceSpinBox.setRange(0.0, 1000000.0);
	m_GridLayout.addWidget(&m_FocalDistanceSpinBox, 6, 2);
	
	connect(&m_FocalDistanceSlider, SIGNAL(valueChanged(double)), &m_FocalDistanceSpinBox, SLOT(setValue(double)));
	connect(&m_FocalDistanceSlider, SIGNAL(valueChanged(double)), this, SLOT(SetFocalDistance(double)));
	connect(&m_FocalDistanceSpinBox, SIGNAL(valueChanged(double)), &m_FocalDistanceSlider, SLOT(setValue(double)));
}

void CFocusWidget::SetFocusType(const int& FocusType)
{
	if (!Scene())
		return;

	Scene()->m_Camera.m_Focus.m_Type = (CFocus::EType)FocusType;

	// Flag the camera as dirty, this will restart the rendering
	Scene()->m_DirtyFlags.SetFlag(CameraDirty);
}

void CFocusWidget::SetFocalDistance(const double& FocalDistance)
{
	if (!Scene())
		return;

	Scene()->m_Camera.m_Focus.m_FocalDistance = FocalDistance;

	// Flag the camera as dirty, this will restart the rendering
	Scene()->m_DirtyFlags.SetFlag(CameraDirty);
}
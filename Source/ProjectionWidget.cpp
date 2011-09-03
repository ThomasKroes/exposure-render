
#include "ProjectionWidget.h"
#include "MainWindow.h"
#include "Scene.h"

CProjectionWidget::CProjectionWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_GridLayout(),
	m_FieldOfViewSlider(),
	m_FieldOfViewSpinBox()
{
	setTitle("Projection");
	setStatusTip("Projection properties");
	setToolTip("Projection properties");

	setLayout(&m_GridLayout);

	// Field of view
	m_GridLayout.addWidget(new QLabel("Field of view"), 4, 0);

	m_FieldOfViewSlider.setOrientation(Qt::Orientation::Horizontal);
	m_FieldOfViewSlider.setRange(10, 200);
	m_GridLayout.addWidget(&m_FieldOfViewSlider, 4, 1);
	
    m_FieldOfViewSpinBox.setRange(10, 200);
	m_GridLayout.addWidget(&m_FieldOfViewSpinBox, 4, 2);
	
	connect(&m_FieldOfViewSlider, SIGNAL(valueChanged(int)), &m_FieldOfViewSpinBox, SLOT(setValue(int)));
	connect(&m_FieldOfViewSlider, SIGNAL(valueChanged(int)), this, SLOT(SetFieldOfView(int)));
	connect(&m_FieldOfViewSpinBox, SIGNAL(valueChanged(int)), &m_FieldOfViewSlider, SLOT(setValue(int)));
}

void CProjectionWidget::SetFieldOfView(const int& FieldOfView)
{
	if (!gpScene)
		return;

	gpScene->m_Camera.m_FovV = FieldOfView;

	// Flag the camera as dirty, this will restart the rendering
	gpScene->m_DirtyFlags.SetFlag(CameraDirty);
}
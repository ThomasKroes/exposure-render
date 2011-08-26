
#include "VtkWidget.h"

CVtkWidget::CVtkWidget(QWidget* pParent) :
	QWidget(pParent),
	m_pQtVtkWidget(NULL),
	m_pMainLayout(NULL),
	m_pTopLayout(NULL),
	m_pBottomLayout(NULL)
{
	// Create main vertical layout
	m_pMainLayout = new QVBoxLayout();
	setLayout(m_pMainLayout);

	// Create VTK widget and add to vertical layout
	m_pQtVtkWidget = new QVTKWidget();
	
	// Create and add top layout
	m_pTopLayout = new QHBoxLayout();
	m_pMainLayout->addLayout(m_pTopLayout);

	// Add VTK widget
	m_pMainLayout->addWidget(m_pQtVtkWidget);

	// Create and add bottom layout
	m_pBottomLayout = new QHBoxLayout();
	m_pMainLayout->addLayout(m_pBottomLayout);

//	m_pBottomLayout->addWidget(NULL);

//	m_pZoomLabel = new QLabel("Zoom");
//	m_pBottomLayout->addWidget(m_pZoomLabel);

//	m_pZoomSlider = new QSlider(Qt::Orientation::Horizontal);
//	m_pBottomLayout->addWidget(m_pZoomSlider);

//	m_pZoomComboBox = new QComboBox("100 %");
//	m_pZoomComboBox->addItem("10 %");
//	m_pBottomLayout->addWidget(m_pZoomComboBox);
}
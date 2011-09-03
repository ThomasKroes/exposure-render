
#include "LoadSettingsDialog.h"

CLoadSettingsDialog::CLoadSettingsDialog(QWidget* pParent) :
	QDialog(pParent),
	m_pMainLayout(NULL),
	m_pResampleGroupBox(NULL),
	m_pResampleLayout(NULL),
	m_pResampleXLabel(NULL),
	m_pResampleXSlider(NULL),
	m_pResampleXSpinBox(NULL),
	m_pResampleYLabel(NULL),
	m_pResampleYSlider(NULL),
	m_pResampleYSpinBox(NULL),
	m_pLockYCheckBox(NULL),
	m_pResampleZLabel(NULL),
	m_pResampleZSlider(NULL),
	m_pResampleZSpinBox(NULL),
	m_pLockZCheckBox(NULL),
	m_pDialogButtons(NULL),
	m_Resample(false),
	m_ResampleX(1.0f),
	m_ResampleY(1.0f),
	m_ResampleZ(1.0f)
{
	resize(400, 100);

	setWindowTitle("Import settings");
	setWindowIcon(QIcon(":/Images/gear.png"));

	// Create main layout
	m_pMainLayout = new QVBoxLayout(this);

	m_pResampleGroupBox = new QGroupBox(this);
	m_pResampleGroupBox->setTitle("Resampling");
	m_pResampleGroupBox->setCheckable(true);
	m_pMainLayout->addWidget(m_pResampleGroupBox);

//	connect(m_pResampleGroupBox, SIGNAL(stateChanged(int)), m_pResampleXSpinBox, SLOT(SetResample(int)));
	
	// Align
	m_pResampleLayout = new QGridLayout(m_pResampleGroupBox);
	m_pResampleLayout->setAlignment(Qt::AlignTop);
	m_pResampleLayout->setColumnStretch(0, 2);
	m_pResampleLayout->setColumnStretch(1, 10);
	m_pResampleLayout->setColumnStretch(2, 2);
	
	// Attach layout to group box
	m_pResampleGroupBox->setLayout(m_pResampleLayout);

	// Scaling x
	m_pResampleXLabel = new QLabel("X Resample (%)");
	m_pResampleLayout->addWidget(m_pResampleXLabel, 0, 0);

	m_pResampleXSlider = new QSlider(Qt::Orientation::Horizontal);
    m_pResampleXSlider->setFocusPolicy(Qt::StrongFocus);
    m_pResampleXSlider->setTickPosition(QSlider::TickPosition::NoTicks);
	m_pResampleXSlider->setRange(0, 100);
	m_pResampleLayout->addWidget(m_pResampleXSlider, 0, 1);
	
	m_pResampleXSpinBox = new QSpinBox;
    m_pResampleXSpinBox->setRange(0, 100);
	m_pResampleLayout->addWidget(m_pResampleXSpinBox, 0, 2);
	
	connect(m_pResampleXSlider, SIGNAL(valueChanged(int)), m_pResampleXSpinBox, SLOT(setValue(int)));
	connect(m_pResampleXSpinBox, SIGNAL(valueChanged(int)), m_pResampleXSlider, SLOT(setValue(int)));
	connect(m_pResampleXSpinBox, SIGNAL(valueChanged(int)), this, SLOT(SetResampleX(int)));

	// Scaling y
	m_pResampleYLabel = new QLabel("Y Resample (%)");
	m_pResampleLayout->addWidget(m_pResampleYLabel, 1, 0);

	m_pResampleYSlider = new QSlider(Qt::Orientation::Horizontal);
    m_pResampleYSlider->setFocusPolicy(Qt::StrongFocus);
    m_pResampleYSlider->setTickPosition(QSlider::TickPosition::NoTicks);
	m_pResampleYSlider->setRange(0, 100);
	m_pResampleLayout->addWidget(m_pResampleYSlider, 1, 1);
	
	m_pResampleYSpinBox = new QSpinBox;
    m_pResampleYSpinBox->setRange(0, 100);
	m_pResampleLayout->addWidget(m_pResampleYSpinBox, 1, 2);
	
	m_pLockYCheckBox = new QCheckBox("Lock", this);
	m_pResampleLayout->addWidget(m_pLockYCheckBox, 1, 3);

	connect(m_pResampleYSlider, SIGNAL(valueChanged(int)), m_pResampleYSpinBox, SLOT(setValue(int)));
	connect(m_pResampleYSpinBox, SIGNAL(valueChanged(int)), m_pResampleYSlider, SLOT(setValue(int)));
	connect(m_pResampleYSpinBox, SIGNAL(valueChanged(int)), this, SLOT(SetResampleY(int)));
	connect(m_pLockYCheckBox, SIGNAL(stateChanged(int)), this, SLOT(LockY(int)));

	// Scaling z
	m_pResampleZLabel = new QLabel("Z Resample (%)");
	m_pResampleLayout->addWidget(m_pResampleZLabel, 2, 0);

	m_pResampleZSlider = new QSlider(Qt::Orientation::Horizontal);
    m_pResampleZSlider->setFocusPolicy(Qt::StrongFocus);
    m_pResampleZSlider->setTickPosition(QSlider::TickPosition::NoTicks);
	m_pResampleZSlider->setRange(0, 100);
	m_pResampleLayout->addWidget(m_pResampleZSlider, 2, 1);
	
	m_pResampleZSpinBox = new QSpinBox;
    m_pResampleZSpinBox->setRange(0, 100);
	
	m_pResampleLayout->addWidget(m_pResampleZSpinBox, 2, 2);
	
	m_pLockZCheckBox = new QCheckBox("Lock", this);
	m_pResampleLayout->addWidget(m_pLockZCheckBox, 2, 3);

	connect(m_pResampleZSlider, SIGNAL(valueChanged(int)), m_pResampleZSpinBox, SLOT(setValue(int)));
	connect(m_pResampleZSpinBox, SIGNAL(valueChanged(int)), m_pResampleZSlider, SLOT(setValue(int)));
	connect(m_pResampleZSlider, SIGNAL(valueChanged(int)), this, SLOT(SetResampleZ(int)));
	connect(m_pLockZCheckBox, SIGNAL(stateChanged(int)), this, SLOT(LockZ(int)));
	
	// Dialog buttons
	m_pDialogButtons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel | QDialogButtonBox::Reset);
	connect(m_pDialogButtons, SIGNAL(accepted()), this, SLOT(Accept()));
    connect(m_pDialogButtons, SIGNAL(rejected()), this, SLOT(Reject()));
	connect(m_pDialogButtons, SIGNAL(clicked(QAbstractButton*)), this, SLOT(Clicked(QAbstractButton*)));

	m_pMainLayout->addWidget(m_pDialogButtons);

	// Reset to defaults
	Reset();

	// Set tooltips
	SetToolTips();
};

bool CLoadSettingsDialog::GetResample(void)		{ return m_Resample;  }
float CLoadSettingsDialog::GetResampleX(void)	{ return m_ResampleX; }
float CLoadSettingsDialog::GetResampleY(void)	{ return m_ResampleY; }
float CLoadSettingsDialog::GetResampleZ(void)	{ return m_ResampleZ; }

void CLoadSettingsDialog::LockY(const int& State)
{
	m_pResampleYLabel->setEnabled(!State);
	m_pResampleYSlider->setEnabled(!State);
	m_pResampleYSpinBox->setEnabled(!State);

	if (State)
	{
		connect(m_pResampleXSlider, SIGNAL(valueChanged(int)), m_pResampleYSlider, SLOT(setValue(int)));
		connect(m_pResampleXSpinBox, SIGNAL(valueChanged(int)), m_pResampleYSpinBox, SLOT(setValue(int)));

		m_pResampleYSlider->setValue(m_pResampleXSlider->value());
	}
	else
	{
		disconnect(m_pResampleXSlider, SIGNAL(valueChanged(int)), m_pResampleYSlider, SLOT(setValue(int)));
		disconnect(m_pResampleXSpinBox, SIGNAL(valueChanged(int)), m_pResampleYSpinBox, SLOT(setValue(int)));
	}
}

void CLoadSettingsDialog::LockZ(const int& State)
{
	m_pResampleZLabel->setEnabled(!State);
	m_pResampleZSlider->setEnabled(!State);
	m_pResampleZSpinBox->setEnabled(!State);

	if (State)
	{
		connect(m_pResampleXSlider, SIGNAL(valueChanged(int)), m_pResampleZSlider, SLOT(setValue(int)));
		connect(m_pResampleXSpinBox, SIGNAL(valueChanged(int)), m_pResampleZSpinBox, SLOT(setValue(int)));

		m_pResampleZSlider->setValue(m_pResampleXSlider->value());
	}
	else
	{
		disconnect(m_pResampleXSlider, SIGNAL(valueChanged(int)), m_pResampleZSlider, SLOT(setValue(int)));
		disconnect(m_pResampleXSpinBox, SIGNAL(valueChanged(int)), m_pResampleZSpinBox, SLOT(setValue(int)));
	}
}

void CLoadSettingsDialog::SetResample(const int& Resample)
{
	m_Resample = Resample;
}

void CLoadSettingsDialog::SetResampleX(const int& ResampleX)
{
	m_ResampleX = 0.01f * (float)ResampleX;
}

void CLoadSettingsDialog::SetResampleY(const int& ResampleY)
{
	m_ResampleY = 0.01f * (float)ResampleY;
}

void CLoadSettingsDialog::SetResampleZ(const int& ResampleZ)
{
	m_ResampleZ = 0.01f * (float)ResampleZ;
}

void CLoadSettingsDialog::Accept(void)
{
	accept();
}

void CLoadSettingsDialog::Reject(void)
{
	reject();
}

void CLoadSettingsDialog::Clicked(QAbstractButton* pButton)
{
	if (pButton->text() == tr("Reset"))
		Reset();
}

void CLoadSettingsDialog::Reset(void)
{
	m_pResampleGroupBox->setChecked(m_Resample);
	m_pResampleXSlider->setValue(m_ResampleX * 100.0f);
	m_pResampleXSpinBox->setValue(m_ResampleX * 100.0f);
	m_pResampleYSlider->setValue(m_ResampleY * 100.0f);
	m_pResampleYSpinBox->setValue(m_ResampleY * 100.0f);
	m_pResampleZSlider->setValue(m_ResampleZ * 100.0f);
	m_pResampleZSpinBox->setValue(m_ResampleZ * 100.0f);
	m_pLockYCheckBox->setChecked(true);
	m_pLockZCheckBox->setChecked(true);
}

void CLoadSettingsDialog::SetToolTips(void)
{
	m_pResampleGroupBox->setToolTip("Resample");
	m_pResampleXLabel->setToolTip("Resample X");
	m_pResampleXSlider->setToolTip("Drag to adjust the X scale");
	m_pResampleXSpinBox->setToolTip("Spin to adjust the X scale");
	m_pResampleYLabel->setToolTip("X scale");
	m_pResampleYSlider->setToolTip("Drag to adjust the Y scale");
	m_pResampleYSpinBox->setToolTip("Spin to adjust the Y scale");
	m_pLockYCheckBox->setToolTip("Lock the Y scale");
	m_pResampleZLabel->setToolTip("Y scale");
	m_pResampleZSlider->setToolTip("Drag to adjust the Z scale");
	m_pResampleZSpinBox->setToolTip("Spin to adjust the Z scale");
	m_pLockZCheckBox->setToolTip("Lock the Z scale");
	m_pDialogButtons->setToolTip("Z scale");
}

// Precompiled headers
#include "Stable.h"

#include "LoadSettingsDialog.h"

QLoadSettingsDialog::QLoadSettingsDialog(QWidget* pParent) :
	QDialog(pParent),
	m_MainLayout(),
	m_ResampleGroupBox(),
	m_ResampleLayout(),
	m_ResampleXLabel(),
	m_ResampleXSlider(),
	m_ResampleXSpinBox(),
	m_ResampleYLabel(),
	m_ResampleYSlider(),
	m_ResampleYSpinBox(),
	m_LockYCheckBox(),
	m_ResampleZLabel(),
	m_ResampleZSlider(),
	m_ResampleZSpinBox(),
	m_LockZCheckBox(),
	m_DialogButtons(),
	m_Resample(false),
	m_ResampleX(1.0f),
	m_ResampleY(1.0f),
	m_ResampleZ(1.0f)
{
	resize(400, 100);

	setWindowTitle("Import settings");
	setWindowIcon(GetIcon("gear"));

	setLayout(&m_MainLayout);

	QSettings Settings;
	
	// Create main layout
	m_ResampleGroupBox.setTitle("Resampling");
	m_ResampleGroupBox.setCheckable(true);
	m_ResampleGroupBox.setChecked(Settings.value("resampling/enabled", false).toBool());
	m_MainLayout.addWidget(&m_ResampleGroupBox);

	// Align
	m_ResampleLayout.setAlignment(Qt::AlignTop);
	m_ResampleLayout.setColumnStretch(0, 2);
	m_ResampleLayout.setColumnStretch(1, 10);
	m_ResampleLayout.setColumnStretch(2, 2);
	
	// Attach layout to group box
	m_ResampleGroupBox.setLayout(&m_ResampleLayout);

	// Scaling x
	m_ResampleXLabel.setText("X Resample (%)");
	m_ResampleLayout.addWidget(&m_ResampleXLabel, 0, 0);

	m_ResampleXSlider.setOrientation(Qt::Horizontal);
    m_ResampleXSlider.setTickPosition(QDoubleSlider::NoTicks);
	m_ResampleXSlider.setRange(0.0, 1.0);
	m_ResampleLayout.addWidget(&m_ResampleXSlider, 0, 1);
	
    m_ResampleXSpinBox.setRange(0.0, 1.0);
	m_ResampleXSpinBox.setSingleStep(0.1);
	m_ResampleLayout.addWidget(&m_ResampleXSpinBox, 0, 2);
	
	connect(&m_ResampleXSlider, SIGNAL(valueChanged(double)), &m_ResampleXSpinBox, SLOT(setValue(double)));
	connect(&m_ResampleXSpinBox, SIGNAL(valueChanged(double)), &m_ResampleXSlider, SLOT(setValue(double)));
	connect(&m_ResampleXSpinBox, SIGNAL(valueChanged(double)), this, SLOT(SetResampleX(double)));

	m_ResampleXSlider.setValue(Settings.value("resampling/x", 1.0).toDouble());

	// Scaling y
	m_ResampleYLabel.setText("Y Resample (%)");
	m_ResampleLayout.addWidget(&m_ResampleYLabel, 1, 0);

	m_ResampleYSlider.setOrientation(Qt::Horizontal);
    m_ResampleYSlider.setTickPosition(QDoubleSlider::NoTicks);
	m_ResampleYSlider.setRange(0.0, 1.0);
	m_ResampleLayout.addWidget(&m_ResampleYSlider, 1, 1);
	
    m_ResampleYSpinBox.setRange(0.0, 1.0);
	m_ResampleLayout.addWidget(&m_ResampleYSpinBox, 1, 2);
	
	m_LockYCheckBox.setText("Lock");
	m_ResampleLayout.addWidget(&m_LockYCheckBox, 1, 3);

	connect(&m_ResampleYSlider, SIGNAL(valueChanged(double)), &m_ResampleYSpinBox, SLOT(setValue(double)));
	connect(&m_ResampleYSpinBox, SIGNAL(valueChanged(double)), &m_ResampleYSlider, SLOT(setValue(double)));
	connect(&m_ResampleYSpinBox, SIGNAL(valueChanged(double)), this, SLOT(SetResampleY(double)));
	connect(&m_LockYCheckBox, SIGNAL(stateChanged(int)), this, SLOT(LockY(int)));

	m_ResampleYSlider.setValue(Settings.value("resampling/y", 1.0).toDouble());
	m_LockYCheckBox.setChecked(Settings.value("resampling/lock-y", true).toBool());

	// Scaling z
	m_ResampleZLabel.setText("Z Resample (%)");
	m_ResampleLayout.addWidget(&m_ResampleZLabel, 2, 0);

	m_ResampleZSlider.setOrientation(Qt::Horizontal);
    m_ResampleZSlider.setTickPosition(QDoubleSlider::NoTicks);
	m_ResampleZSlider.setRange(0.0, 1.0);
	m_ResampleLayout.addWidget(&m_ResampleZSlider, 2, 1);
	
    m_ResampleZSpinBox.setRange(0.0, 1.0);
	
	m_ResampleLayout.addWidget(&m_ResampleZSpinBox, 2, 2);
	
	m_LockZCheckBox.setText("Lock");
	m_ResampleLayout.addWidget(&m_LockZCheckBox, 2, 3);

	connect(&m_ResampleZSlider, SIGNAL(valueChanged(double)), &m_ResampleZSpinBox, SLOT(setValue(double)));
	connect(&m_ResampleZSpinBox, SIGNAL(valueChanged(double)), &m_ResampleZSlider, SLOT(setValue(double)));
	connect(&m_ResampleZSlider, SIGNAL(valueChanged(double)), this, SLOT(SetResampleZ(double)));
	connect(&m_LockZCheckBox, SIGNAL(stateChanged(int)), this, SLOT(LockZ(int)));
	
	m_ResampleZSlider.setValue(Settings.value("resampling/z", 1.0).toDouble());
	m_LockZCheckBox.setChecked(Settings.value("resampling/lock-z", true).toBool());

	// Dialog buttons
	m_DialogButtons.setStandardButtons(QDialogButtonBox::Ok | QDialogButtonBox::Cancel | QDialogButtonBox::Reset);
	connect(&m_DialogButtons, SIGNAL(accepted()), this, SLOT(Accept()));
    connect(&m_DialogButtons, SIGNAL(rejected()), this, SLOT(Reject()));
	connect(&m_DialogButtons, SIGNAL(clicked(QAbstractButton*)), this, SLOT(Clicked(QAbstractButton*)));

	m_MainLayout.addWidget(&m_DialogButtons);

	// Set tooltips
	SetToolTips();
};

bool QLoadSettingsDialog::GetResample(void)		{ return m_Resample;  }
float QLoadSettingsDialog::GetResampleX(void)	{ return m_ResampleX; }
float QLoadSettingsDialog::GetResampleY(void)	{ return m_ResampleY; }
float QLoadSettingsDialog::GetResampleZ(void)	{ return m_ResampleZ; }

void QLoadSettingsDialog::LockY(const int& State)
{
	m_ResampleYLabel.setEnabled(!State);
	m_ResampleYSlider.setEnabled(!State);
	m_ResampleYSpinBox.setEnabled(!State);

	if (State)
	{
		connect(&m_ResampleXSlider, SIGNAL(valueChanged(double)), &m_ResampleYSlider, SLOT(setValue(double)));
		connect(&m_ResampleXSpinBox, SIGNAL(valueChanged(double)), &m_ResampleYSpinBox, SLOT(setValue(double)));

		m_ResampleYSlider.setValue(m_ResampleXSlider.value());
	}
	else
	{
		disconnect(&m_ResampleXSlider, SIGNAL(valueChanged(double)), &m_ResampleYSlider, SLOT(setValue(double)));
		disconnect(&m_ResampleXSpinBox, SIGNAL(valueChanged(double)), &m_ResampleYSpinBox, SLOT(setValue(double)));
	}
}

void QLoadSettingsDialog::LockZ(const int& State)
{
	m_ResampleZLabel.setEnabled(!State);
	m_ResampleZSlider.setEnabled(!State);
	m_ResampleZSpinBox.setEnabled(!State);

	if (State)
	{
		connect(&m_ResampleXSlider, SIGNAL(valueChanged(double)), &m_ResampleZSlider, SLOT(setValue(double)));
		connect(&m_ResampleXSpinBox, SIGNAL(valueChanged(double)), &m_ResampleZSpinBox, SLOT(setValue(double)));

		m_ResampleZSlider.setValue(m_ResampleXSlider.value());
	}
	else
	{
		disconnect(&m_ResampleXSlider, SIGNAL(valueChanged(double)), &m_ResampleZSlider, SLOT(setValue(double)));
		disconnect(&m_ResampleXSpinBox, SIGNAL(valueChanged(double)), &m_ResampleZSpinBox, SLOT(setValue(double)));
	}
}

void QLoadSettingsDialog::SetResample(const int& Resample)
{
	m_Resample = Resample;
}

void QLoadSettingsDialog::SetResampleX(const double& ResampleX)
{
	m_ResampleX = (float)ResampleX;
}

void QLoadSettingsDialog::SetResampleY(const double& ResampleY)
{
	m_ResampleY = (float)ResampleY;
}

void QLoadSettingsDialog::SetResampleZ(const double& ResampleZ)
{
	m_ResampleZ = (float)ResampleZ;
}

void QLoadSettingsDialog::Accept(void)
{
	accept();
}

void QLoadSettingsDialog::Reject(void)
{
	reject();
}

void QLoadSettingsDialog::Clicked(QAbstractButton* pButton)
{
	if (pButton->text() == tr("Reset"))
		Reset();
}

void QLoadSettingsDialog::Reset(void)
{
	m_ResampleGroupBox.setChecked(m_Resample);
	m_ResampleXSlider.setValue(m_ResampleX * 100.0f);
	m_ResampleXSpinBox.setValue(m_ResampleX * 100.0f);
	m_ResampleYSlider.setValue(m_ResampleY * 100.0f);
	m_ResampleYSpinBox.setValue(m_ResampleY * 100.0f);
	m_ResampleZSlider.setValue(m_ResampleZ * 100.0f);
	m_ResampleZSpinBox.setValue(m_ResampleZ * 100.0f);
	m_LockYCheckBox.setChecked(true);
	m_LockZCheckBox.setChecked(true);
}

void QLoadSettingsDialog::SetToolTips(void)
{
	m_ResampleGroupBox.setToolTip("Resample");
	m_ResampleXLabel.setToolTip("Resample X");
	m_ResampleXSlider.setToolTip("Drag to adjust the X scale");
	m_ResampleXSpinBox.setToolTip("Spin to adjust the X scale");
	m_ResampleYLabel.setToolTip("X scale");
	m_ResampleYSlider.setToolTip("Drag to adjust the Y scale");
	m_ResampleYSpinBox.setToolTip("Spin to adjust the Y scale");
	m_LockYCheckBox.setToolTip("Lock the Y scale");
	m_ResampleZLabel.setToolTip("Y scale");
	m_ResampleZSlider.setToolTip("Drag to adjust the Z scale");
	m_ResampleZSpinBox.setToolTip("Spin to adjust the Z scale");
	m_LockZCheckBox.setToolTip("Lock the Z scale");
	m_DialogButtons.setToolTip("Z scale");
}

void QLoadSettingsDialog::accept()
{
	QSettings Settings;

	Settings.setValue("resampling/enabled", m_ResampleGroupBox.isChecked());
	Settings.setValue("resampling/x", m_ResampleXSlider.value());
	Settings.setValue("resampling/y", m_ResampleYSlider.value());
	Settings.setValue("resampling/lock-y", m_LockYCheckBox.isEnabled());
	Settings.setValue("resampling/z", m_ResampleZSlider.value());
	Settings.setValue("resampling/lock-z", m_LockZCheckBox.isEnabled());

	QDialog::accept();
}
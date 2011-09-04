#pragma once

#include <QtGui>

class CLoadSettingsDialog : public QDialog
{
	Q_OBJECT

public:
    CLoadSettingsDialog(QWidget* pParent = NULL);

public:
	bool GetResample(void);
	float GetResampleX(void);
	float GetResampleY(void);
	float GetResampleZ(void);

	void Reset(void);
	void SetToolTips(void);

private slots:
	void LockY(const int& State);
	void LockZ(const int& State);
	void SetResample(const int& Resample);
	void SetResampleX(const int& ResampleX);
	void SetResampleY(const int& ResampleY);
	void SetResampleZ(const int& ResampleZ);
	void Accept(void);
	void Reject(void);
	void Clicked(QAbstractButton* pButton);

private:
	QGridLayout			m_MainLayout;
	QGroupBox			m_ResampleGroupBox;
	QGridLayout			m_ResampleLayout;
	QLabel				m_ResampleXLabel;
	QSlider				m_ResampleXSlider;
	QSpinBox			m_ResampleXSpinBox;
	QLabel				m_ResampleYLabel;
	QSlider				m_ResampleYSlider;
	QSpinBox			m_ResampleYSpinBox;
	QCheckBox			m_LockYCheckBox;
	QLabel				m_ResampleZLabel;
	QSlider				m_ResampleZSlider;
	QSpinBox			m_ResampleZSpinBox;
	QCheckBox			m_LockZCheckBox;
	QDialogButtonBox	m_DialogButtons;

	bool				m_Resample;
	float				m_ResampleX;
	float				m_ResampleY;
	float				m_ResampleZ;
};
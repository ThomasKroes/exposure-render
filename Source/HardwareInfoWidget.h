#pragma once

class QHardwareInfoWidget : public QGroupBox
{
    Q_OBJECT

public:
    QHardwareInfoWidget(QWidget* pParent = NULL);

private slots:

private:
	QGridLayout		m_MainLayout;
};
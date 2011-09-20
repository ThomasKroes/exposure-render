#pragma once

class QAboutDialog : public QDialog
{
	Q_OBJECT

public:
	QAboutDialog(QWidget* pParent = NULL);

private:
	QGridLayout			m_MainLayout;
	QDialogButtonBox	m_DialogButtons;
};
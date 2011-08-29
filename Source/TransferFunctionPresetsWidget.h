#pragma once

#include <QtGui>

#include "TransferFunction.h"

class CVolumeAppearanceDockWidget;
class QTransferFunctionWidget;

class QTransferFunctionGradientWidget : public QPushButton
{
    Q_OBJECT

public:
    QTransferFunctionGradientWidget(QWidget* pParent = NULL, QTransferFunction* pTransferFunction = NULL) :
		QPushButton(pParent),
		m_pTransferFunction(pTransferFunction)
	{
	}
 
virtual void paintEvent(QPaintEvent *);
	QTransferFunction*	m_pTransferFunction;
};

class QTransferFunctionPresetsModel : public QAbstractListModel
 {
     Q_OBJECT

 public:
     QTransferFunctionPresetsModel(QObject* pParent = NULL);

	 int rowCount(const QModelIndex &parent = QModelIndex()) const { return m_TransferFunctions.size(); }
	 virtual int columnCount(const QModelIndex &parent = QModelIndex()) const { return 1; };

	 QVariant data(const QModelIndex& Index, int Role) const;

	 QVariant headerData(int Section, Qt::Orientation Orientation, int Role) const
	 {
		 return QVariant();

		if (Role != Qt::DisplayRole)
			return QVariant();

		if (Orientation == Qt::Horizontal)
		{
			switch (Section)
			{
				case 0: return "Name";
				case 1: return "Function";
			}
		}
	 }


     QList<QTransferFunction>	m_TransferFunctions;
 };

class QTransferFunctionPresetsWidget : public QGroupBox
{
    Q_OBJECT

public:
    QTransferFunctionPresetsWidget(QWidget* pParent = NULL);

private:

protected:
	QGridLayout*		m_pGridLayout;
	QLabel*				m_pNameLabel;
	QLineEdit*			m_pPresetNameEdit;
	QPushButton*		m_pLoadPresetsPushButton;
	QPushButton*		m_pSavePresetPushButton;
	QPushButton*		m_pSavePresetsPushButton;
	QPushButton*		m_pRemovePresetPushButton;
	QListWidget*		m_pTable;
	QStandardItemModel	m_Model;
};
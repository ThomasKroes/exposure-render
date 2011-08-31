#pragma once

#include <QtGui>

class QStatisticsWidget : public QWidget
{
    Q_OBJECT

public:
    QStatisticsWidget(QWidget* pParent = NULL);

	void Init(void);

private:
	void PopulateTree(void);
	QTreeWidgetItem* AddItem(QTreeWidgetItem* pParent, const QString& Property, const QString& Value = "", const QString& Unit = "");
	void UpdateStatistic(const QString& Property, const QString& Value);

public slots:
	void OnRenderBegin(void);
	void OnRenderEnd(void);
	void OnMemoryAllocate(void);
	void OnMemoryFree(void);
	void OnPreFrame(void);
	void OnPostFrame(void);

private:
	QGridLayout		m_MainLayout;
	QGroupBox		m_Group;
	QGridLayout		m_GroupLayout;
	QTreeWidget		m_Tree;
};
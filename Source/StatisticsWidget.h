#pragma once

#include <QtGui>

class QStatisticsWidget : public QWidget
{
    Q_OBJECT

public:
    QStatisticsWidget(QWidget* pParent = NULL);

	QSize sizeHint() const;
	void Init(void);

private:
	void PopulateTree(void);
	QTreeWidgetItem* AddItem(QTreeWidgetItem* pParent, const QString& Property, const QString& Value = "", const QString& Unit = "");
	void UpdateStatistic(const QString& Property, const QString& Value);
	QTreeWidgetItem* FindItem(const QString& Name);

public slots:
	void OnRenderBegin(void);
	void OnRenderEnd(void);
	void OnPreRenderFrame(void);
	void OnPostRenderFrame(void);
	void OnBufferSizeChanged(const QString& Name, const int& Size);
	void OnExpandAll(const bool& Expand);
	
private:
	QGridLayout		m_MainLayout;
	QGroupBox		m_Group;
	QGridLayout		m_GroupLayout;
	QTreeWidget		m_Tree;
};
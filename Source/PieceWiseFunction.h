#pragma once

#include <QtGui>

class CPieceWiseFunction : public QGraphicsPolygonItem
{
public:
    CPieceWiseFunction()
	{
		
	};

	void paint(QPainter* pPainter, const QStyleOptionGraphicsItem* pOption, QWidget* )
	{
		
		QGraphicsPolygonItem::paint(pPainter, pOption);
	}


protected:
//	QList<CNode *>			m_Nodes;
};
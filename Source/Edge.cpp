
#include "Edge.h"

int				CEdge::m_NoInstances	= 0;
float			CEdge::m_PenWidth		= 1.2f;
float			CEdge::m_PenWidthHover	= 1.2f;
QColor			CEdge::m_PenColor		= QColor(60, 60, 60);
QColor			CEdge::m_PenColorHover	= QColor(220, 220, 220);
Qt::PenStyle	CEdge::m_PenStyle		= Qt::PenStyle::SolidLine;
Qt::PenStyle	CEdge::m_PenStyleHover	= Qt::PenStyle::DashLine;
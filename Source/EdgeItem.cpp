
#include "NodeItem.h"
#include "TransferFunction.h"
#include "TransferFunctionCanvas.h"

QEdgeItem::QEdgeItem(QTransferFunctionCanvas* pTransferFunctionCanvas) :
	QGraphicsLineItem(pTransferFunctionCanvas),
	m_pTransferFunctionCanvas(pTransferFunctionCanvas)
{
};
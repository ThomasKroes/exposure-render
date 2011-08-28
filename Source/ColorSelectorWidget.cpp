
#include "ColorSelectorWidget.h"

/*
class QColorShowLabel : public QFrame
{
    Q_OBJECT

public:
    QColorShowLabel(QWidget *parent) : QFrame(parent) {
        setFrameStyle(QFrame::Panel|QFrame::Sunken);
        setAcceptDrops(true);
        mousePressed = false;
    }
    void setColor(QColor c) { col = c; }

signals:
    void colorDropped(QRgb);

protected:
    void paintEvent(QPaintEvent *);
    void mousePressEvent(QMouseEvent *e);
    void mouseMoveEvent(QMouseEvent *e);
    void mouseReleaseEvent(QMouseEvent *e);
#ifndef QT_NO_DRAGANDDROP
    void dragEnterEvent(QDragEnterEvent *e);
    void dragLeaveEvent(QDragLeaveEvent *e);
    void dropEvent(QDropEvent *e);
#endif

private:
    QColor col;
    bool mousePressed;
    QPoint pressPos;
};

void QColorShowLabel::paintEvent(QPaintEvent *e)
{
    QPainter p(this);
    drawFrame(&p);
    p.fillRect(contentsRect()&e->rect(), col);
}

void QColorShowLabel::mousePressEvent(QMouseEvent *e)
{
    mousePressed = true;
    pressPos = e->pos();
}

void QColorShowLabel::mouseMoveEvent(QMouseEvent *e)
{
#ifdef QT_NO_DRAGANDDROP
    Q_UNUSED(e);
#else
    if (!mousePressed)
        return;
    if ((pressPos - e->pos()).manhattanLength() > QApplication::startDragDistance()) {
        QMimeData *mime = new QMimeData;
        mime->setColorData(col);
        QPixmap pix(30, 20);
        pix.fill(col);
        QPainter p(&pix);
        p.drawRect(0, 0, pix.width() - 1, pix.height() - 1);
        p.end();
        QDrag *drg = new QDrag(this);
        drg->setMimeData(mime);
        drg->setPixmap(pix);
        mousePressed = false;
        drg->start();
    }
#endif
}


void QColorShowLabel::dragEnterEvent(QDragEnterEvent *e)
{
    if (qvariant_cast<QColor>(e->mimeData()->colorData()).isValid())
        e->accept();
    else
        e->ignore();
}

void QColorShowLabel::dragLeaveEvent(QDragLeaveEvent *)
{
}

void QColorShowLabel::dropEvent(QDropEvent *e)
{
    QColor color = qvariant_cast<QColor>(e->mimeData()->colorData());
    if (color.isValid()) {
        col = color;
        repaint();
        emit colorDropped(col.rgb());
        e->accept();
    } else {
        e->ignore();
    }
}
*/

QColorFavoritesWidget::QColorFavoritesWidget(QWidget* pParent) :
	QWidget(pParent),
	m_pMainLayout(NULL),
	m_pTableWidget(NULL)
{
	setStatusTip("Favorites");
	setToolTip("Favorites");

	// Main layout
	m_pMainLayout = new QGridLayout();
	m_pMainLayout->setAlignment(Qt::AlignTop);
	setLayout(m_pMainLayout);

	// Create table widget and it to the grid layout
	m_pTableWidget = new QTableWidget;
	m_pTableWidget->setColumnCount(2);
	m_pMainLayout->addWidget(m_pTableWidget, 0, 0);
}

QColorPresetsWidget::QColorPresetsWidget(QWidget* pParent) :
	QWidget(pParent),
	m_pMainLayout(NULL),
	m_pTableWidget(NULL)
{
	setStatusTip("Presets");
	setToolTip("Presets");

	// Main layout
	m_pMainLayout = new QGridLayout();
	m_pMainLayout->setAlignment(Qt::AlignTop);
	setLayout(m_pMainLayout);

	// Create table widget and it to the grid layout
	m_pTableWidget = new QTableWidget;
	m_pTableWidget->setColumnCount(2);
	m_pMainLayout->addWidget(m_pTableWidget, 0, 0);
}

QColorSelectorWidget::QColorSelectorWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_pMainLayout(NULL),
	m_pColorsTab()
{
	setTitle("Color");
	setToolTip("Color settings");

	// Node properties layout
	m_pMainLayout = new QGridLayout();
	m_pMainLayout->setAlignment(Qt::AlignTop);
//	m_pMainLayout->setContentsMargins(0, 0, 0, 0);

	setLayout(m_pMainLayout);

	// Node selection
	m_pColorsTab = new QTabWidget();
	m_pColorsTab->setFixedWidth(200);
	m_pColorsTab->setStatusTip("Color selection");
	m_pColorsTab->setToolTip("Color selection");
	m_pMainLayout->addWidget(m_pColorsTab, 0, 0);

	m_pColorsTab->addTab(new QColorPresetsWidget, "Presets");
	m_pColorsTab->addTab(new QColorFavoritesWidget, "Favorites");

//	setFixedHeight(250);

	// Create main layout
//	m_pMainLayout = new QGridLayout();
//	m_pMainLayout->setAlignment(Qt::AlignTop);
//	setLayout(m_pMainLayout);

//	m_pMainLayout->addWidget(new QColorShowLabel(this));
}
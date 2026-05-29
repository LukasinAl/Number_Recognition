from PyQt6.QtGui import QMouseEvent, QPaintEvent, QPainter, QPen, QImage
from PyQt6.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget
from PyQt6.QtCore import Qt, QLineF
from network_core import Net
import math

class Drawer(QWidget):
  def __init__(self):
    super().__init__()
    self.button = QPushButton("What Number", self)
    self.button.clicked.connect(self.categorize_number)
    self.points = []
    self.connected = []
    self.active = False
    self.penwidth = 4
    self.setFixedSize(784, 784)

    self.net = Net(4, [784, 500, 128, 10])
    self.net.SetActivationRelu()
    self.net.SetLossMSE()
    self.net.ReadFromBinary("../weights/weights.bin")

  
  def mousePressEvent(self, event) -> None:
    if (event.button() == Qt.MouseButton.LeftButton):
      self.points.append(event.position())
      self.active = True
    self.update()

  def mouseMoveEvent(self, event):
    if self.active:
      self.points.append(event.position())
    self.update()
  
  def mouseReleaseEvent(self, event):
    if self.active and event.button() == Qt.MouseButton.LeftButton:
      self.active = False
    self.update()

  def paintEvent(self, event):
    painter = QPainter(self)
    pen = QPen()
    pen.setColor(Qt.GlobalColor.black)
    pen.setWidth(self.penwidth)
    painter.setPen(pen)
    for i in range(len(self.points)):
      painter.drawPoint(self.points[i])

  def categorize_number(self):
    coef_h = self.height() / 28
    coef_w = self.width() / 28
    grid = []
    for i in range(28):
      grid.append([])
      for j in range(28):
        grid[i].append(0.0)
    for i in self.points:
      grid[math.floor(i.x() / coef_w)][math.floor(i.y() / coef_h)] += 1
    
    max_val = max(max(row) for row in grid)
    if max_val == 0:
        max_val = 1 
    print(max_val)
    flattened = []
    for i in grid:
      for j in i:
        j /= max_val
        flattened.append(j)
    print(max(flattened))
    result = self.net.ForwardPass(flattened)
    print(result)




class MainWindow(QMainWindow):
  def __init__(self ):
    super().__init__()

    self.setWindowTitle("Number Recognition")
    self.setFixedSize(600, 600)
    self.paint = Drawer()
    self.setCentralWidget(self.paint)

app = QApplication([])
window = MainWindow()
window.show()
app.exec()
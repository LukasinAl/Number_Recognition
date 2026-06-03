from PyQt6.QtGui import QMouseEvent, QPaintEvent, QPainter, QPen, QImage
from PyQt6.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QVBoxLayout, QTextEdit
from PyQt6.QtCore import Qt, QLineF
from network_core import Net, ActivationFunction
import math

class Drawer(QWidget):
  def __init__(self, TextField=None):
    super().__init__()
    self.grid = [[0 for j in range(28)] for i in range(28)]
    self.active = False
    self.penwidth = 4
    self.setMinimumSize(280, 280)
    self.text = TextField

    self.net = Net(4, [784, 500, 128, 10])
    self.net.SetLayersAcivation([ActivationFunction.RELU, ActivationFunction.RELU, ActivationFunction.SIGMOID])
    self.net.SetLossMSE()
    self.net.ReadFromBinary("weights/weights2.bin")


  def clear_board(self):
    self.grid = [[0 for j in range(28)] for i in range(28)]
    self.update()

  def mousePressEvent(self, event) -> None:
    if (event.button() == Qt.MouseButton.LeftButton):
      if (event.position().x() < 0 or
        event.position().y() < 0 or
        event.position().x() > self.width() or
        event.position().y() > self.height()):
        return
      self.grid[math.floor(event.position().y() * 28 / self.height())][math.floor(event.position().x() * 28 / self.width())] += 1
      self.active = True
    self.update()

  def mouseMoveEvent(self, event):
    if self.active:
      if (event.position().x() < 0 or
          event.position().y() < 0 or
          event.position().x() >= self.width() or
          event.position().y() >= self.height()):
        return
      self.grid[math.floor(event.position().y() * 28 / self.height())][math.floor(event.position().x() * 28 / self.width())] += 1
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
    for i in range(len(self.grid)):
      for j in range(len(self.grid[i])):
        if self.grid[i][j] > 0: 
          painter.drawRect(math.floor(self.width() * j / 28),
                           math.floor(self.height() * i / 28),
                           math.floor(self.width() / 28),
                           math.floor(self.height() / 28))

  def categorize_number(self):
    max_val = max(max(row) for row in self.grid)
    if max_val == 0:
        max_val = 1 
    flattened = []
    for i in range(len(self.grid)):
      for j in range(len(self.grid[i])):
        if self.grid[j][i] > 0:
          flattened.append(1)
        else:
          flattened.append(0)
    result = self.net.ForwardPass(flattened)
    maximum = max(result)
    ans = 0
    for i in range(len(result)):
      if maximum == result[i]:
        ans = i
      print(f"{i} - {result[i]:.2f}")
    self.text.setText(f"Recognized number {ans}")




class MainWindow(QMainWindow):
  def __init__(self ):
    super().__init__()
    self.setWindowTitle("Number Recognition")
    self.setMinimumSize(800, 800)

    self.display = QTextEdit()
    self.paint = Drawer(self.display)

    self.clear_button = QPushButton("Clear", self)
    self.button = QPushButton("What Number", self)
    self.button.clicked.connect(self.paint.categorize_number)
    self.clear_button.clicked.connect(self.paint.clear_board)
    self.display.setReadOnly(True)
    self.display.setMaximumHeight(100)
    self.display.setFontPointSize(20)

    centralWidget = QWidget()
    self.setCentralWidget(centralWidget)
    
    layout = QVBoxLayout(centralWidget)
    layout.addWidget(self.paint)
    layout.addWidget(self.clear_button)
    layout.addWidget(self.button)
    layout.addWidget(self.display)

app = QApplication([])
window = MainWindow()
window.show()
app.exec()
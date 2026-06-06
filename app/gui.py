from PyQt6.QtGui import QMouseEvent, QPaintEvent, QPainter, QPen, QImage
from PyQt6.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QVBoxLayout, QTextEdit, QHBoxLayout, QComboBox
from PyQt6.QtCore import Qt, QLineF
from network_core import Net, ActivationFunction
import math
import csv
import os
import pathlib

script_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(script_dir, "../weights", "weights2.bin")
save_path = os.path.join(script_dir, "handwritten_data.csv")

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
    self.net.ReadFromBinary(weights_path)


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
    self.input = QComboBox()
    self.paint = Drawer(self.display)

    self.clear_button = QPushButton("Clear", self)
    self.classification_button = QPushButton("What Number", self)
    self.save_button = QPushButton("Save sample")
    self.classification_button.clicked.connect(self.paint.categorize_number)
    self.clear_button.clicked.connect(self.paint.clear_board)
    self.save_button.clicked.connect(self.save)

    self.display.setReadOnly(True)
    self.display.setMaximumHeight(100)
    self.display.setFontPointSize(20)
    self.input.setMaximumHeight(20)
    self.input.setMaximumWidth(100)
    self.input.setMaxVisibleItems(10)
    self.input.addItems([str(i) for i in range(10)])

    centralWidget = QWidget()
    self.setCentralWidget(centralWidget)
    
    layout = QVBoxLayout(centralWidget)
    layout.addWidget(self.paint)
    button_layout = QHBoxLayout()
    button_layout.addWidget(self.clear_button)
    button_layout.addWidget(self.classification_button)
    button_layout.addWidget(self.save_button)
    button_layout.addWidget(self.input)
    layout.addLayout(button_layout)
    layout.addWidget(self.display)
  
  def save(self):
    with open(save_path, "a", newline="") as file:
      writer = csv.writer(file)
      flattened = [self.input.currentIndex()]
      for i in range(len(self.paint.grid)):
        for j in range(len(self.paint.grid[i])):
          if self.paint.grid[j][i] > 0:
            flattened.append(1)
          else:
            flattened.append(0)
      writer.writerow(flattened)

app = QApplication([])
app.setStyle("Windows")
window = MainWindow()
window.show()
app.exec()
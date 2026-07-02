from PyQt6.QtGui import QMouseEvent, QPaintEvent, QPainter, QPen, QImage
from PyQt6.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QVBoxLayout, QTextEdit, QHBoxLayout, QComboBox
from PyQt6.QtCore import Qt, QLineF
from network_core import Net, ActivationFunction, LossFunction
import math
import csv
import os
import pathlib
from collections import deque

script_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(script_dir, "../weights", "FinalWeighs.bin")
save_path = os.path.join(script_dir, "handwritten_data.csv")

class Drawer(QWidget):
  """Class providng drawind interface by displaying canvas and tracking mouse events"""
  def __init__(self, TextField : QTextEdit) -> None:
    super().__init__()
    self.grid_scale_x = 140 ## X resolution of drawing canvas
    self.grid_scale_y = 120 ## Y resolution of drawingg canvas
    self.grid = [[0 for j in range(self.grid_scale_x)] for i in range(self.grid_scale_y)]
    ## 0,0 is in left top corner. Y goes as first dimention in grid, X as second
    self.prev = None ## Previous registed event when mouse is active
    self.active = False ## Left button is clicked used to track motion
    self.penwidth = 4
    self.setMinimumSize(280, 280)
    self.text = TextField ## Text field used to display information

    self.net = Net(4, [784, 500, 128, 10])
    self.net.ReadFromBinary(weights_path)


  def clear_board(self) -> None:
    """
    Method which clears board by deleting all registed events,
    clearing history and reloading canvas.
    Method is connected to clear button
    """
    self.grid = [[0 for j in range(self.grid_scale_x)] for i in range(self.grid_scale_y)]
    self.prev = None
    self.update()

  def mousePressEvent(self, event) -> None:
    """
    Method which tracks mouse clicks
    Only Left button is tracked.
    Checks if location of event is within borders, than adds registred event.
    Activates mouse tracking event.
    """
    if (event.button() == Qt.MouseButton.LeftButton):
      if (event.position().x() < 0 or
        event.position().y() < 0 or
        event.position().x() > self.width() or
        event.position().y() > self.height()):
        return
      y = math.floor(event.position().y() * self.grid_scale_y / self.height())
      x = math.floor(event.position().x() * self.grid_scale_x / self.width())
      self.grid[y][x] += 1
      self.prev = (y, x)
      self.active = True
    self.update()

  def mouseMoveEvent(self, event) -> None:
    """
    Method which tracks mouse movement when left button is pressed
    When locations changes calculates it's path between dots by filling line between two points with events.
    It helps to avoid gaps between events when moving mouse fast
    """
    if self.active:
      if (event.position().x() < 0 or
          event.position().y() < 0 or
          event.position().x() >= self.width() or
          event.position().y() >= self.height()):
        return
      y = math.floor(event.position().y() * self.grid_scale_y / self.height())
      x = math.floor(event.position().x() * self.grid_scale_x / self.width())
      (self.grid[y][x]) += 1
      if self.prev:
        y_dist = abs(self.prev[0] - y)
        x_dist = abs(self.prev[1] - x)
        y_start = self.prev[0]
        x_start = self.prev[1]

        ## Calculate Y direction
        if y_dist == 0:
          vec_y = 0
        else:
          vec_y = (y - self.prev[0]) // y_dist
        
        ## Calculate X direction
        if x_dist == 0:
          vec_x = 0
        else:
          vec_x = (x - self.prev[1]) // x_dist

        if x_dist >= y_dist:
          for i in range(x_dist):
            self.grid[y_start + math.floor(y_dist / x_dist * i) * vec_y][x_start + i * vec_x] += 1
        if y_dist > x_dist:
          for i in range(y_dist):
            self.grid[y_start + i * vec_y][x_start + math.floor(x_dist / y_dist * i) * vec_x] += 1
      self.prev = (y, x)
    self.update()
  
  def mouseReleaseEvent(self, event) -> None:
    """
    Method which registres left button release and deactivates mouse tracking.
    """
    if self.active and event.button() == Qt.MouseButton.LeftButton:
      self.active = False
    self.update()

  def paintEvent(self, event) -> None:
    """
    Method which draws whole picture of registred events by drawing recatngles.
    """
    painter = QPainter(self)
    pen = QPen()
    pen.setColor(Qt.GlobalColor.black)
    pen.setWidth(self.penwidth)
    painter.setPen(pen)
    for i in range(len(self.grid)):
      for j in range(len(self.grid[i])):
        if self.grid[i][j] > 0: 
          painter.drawRect(math.floor(self.width() * j / self.grid_scale_x),
                           math.floor(self.height() * i / self.grid_scale_y),
                           math.floor(self.width() / self.grid_scale_x),
                           math.floor(self.height() / self.grid_scale_y))

  def process_digit(self, original_grid : list[list[int]], grid_width : int, grid_height : int) -> int:
    """
    Method which categorixes single handwriten digit

    Args:
      original_grid : 2d list of regisred events
      grid_width : int - width of given list
      grid_geight : int - height of given list

    Returns:
      int : number from 0..9 if categorization was successfull and -1 if not.

    Description:
      Frams exact rectangle where whole digit iis located, extend it to square,
      scales it to fit in c * c grid, adds paddiing and passes it into layerd net,
      returns answer.
    """
    c = 24

    min_width = grid_width
    max_width = 0
    min_height = grid_height
    max_height = 0
    activated = []
    for y in range(len(original_grid)):
      for x in range(len(original_grid[y])):
        if (original_grid[y][x] > 0):
          ## Locate Box containg image and flatenn activated secors
          min_width = min(min_width, x)
          min_height = min(min_height, y)
          max_width = max(max_width, x)
          max_height = max(max_height, y)
          activated.append([y, x])
    diff = (28 - c) // 2
    x_scale = (max_width - min_width) / c
    y_scale = (max_height - min_height) / c
    scale = max(x_scale, y_scale)
    if scale == 0:
      return -1
    x_offset = diff
    y_offset = diff
    if y_scale > x_scale:
      x_offset += math.floor((1 - x_scale / y_scale) * c // 2)
    if x_scale > y_scale:
      y_offset += math.floor((1 - y_scale / x_scale) * c // 2)

    new_grid = [[0 for i in range(28)] for j in range(28)]
    for i in activated:
      (new_grid[y_offset + math.floor((i[0] - min_height) / scale)]
               [x_offset + math.floor((i[1] - min_width) / scale)]) += 1

    flattened = [] ##flatten the array to feed in Neural Network
    for i in range(len(new_grid[0])):
      for j in range(len(new_grid)):
        if new_grid[j][i] > 0:
          flattened.append(1)
        else:
          flattened.append(0)
    result = self.net.ForwardPass(flattened)
    #print(flattened)
    maximum = max(result)
    ans = 0
    for i in range(len(result)):
      if maximum == result[i]:
        ans = i
      #print(f"{i} - {result[i]:.2f}")
    #self.text.setText(f"Recognized number {ans}")

    return ans

  def categorize_digit(self) -> None:
    """
    Method which could be attached to button to categorize digit and display result
    """
    ans = self.process_digit(self.grid, self.grid_scale_x, self.grid_scale_y)
    if ans == -1:
      return
    self.text.setText(f"Number {ans}")

  def categorize_number(self) -> None:
    """
    Method to categorze multi digit number

    description:
      Implemets left to right scan with enumerating all shapes by BFS.
      Feeds each shape to categorizer and merges all result into single number, which is displayed in GUI
    """
    enumeration = [[0 for j in range(self.grid_scale_x)] for i in range(self.grid_scale_y)]
    count = 1
    for j in range(len(self.grid[0])):
      for i in range(len(self.grid)):
        if self.grid[i][j] == 0:
          continue
        res = BFS(self.grid, enumeration, i, j, count)
        if res > 0:
          count += 1
    ans = ""
    for cnt in range(1, count):
      copy_grid = [[0 for j in range(self.grid_scale_x)] for i in range(self.grid_scale_y)]
      for i in range(len(self.grid)):
        for j in range(len(self.grid[i])):
          if self.grid[i][j] == 0 or enumeration[i][j] != cnt:
            continue
          copy_grid[i][j] = self.grid[i][j]
      ans += str(self.process_digit(copy_grid, self.grid_scale_x, self.grid_scale_y))
    self.text.setText(ans)

def BFS(grid : list[list[int]], enumeration : list, y_start : int, x_start : int, num : int) -> int:
  """
  Helper function which is used to separate digits on image
  by performing BFS from given start point and marking all visited dots with given number
  
  Args:
    grid : 2d list of press counts in each sector. Is NOT modified in funtion;
    enumeration : 2d list of same dimetions as grid, is used to store enumeration of image parts;
    y_start : int y coordinate of starting point of BFS should be non negative, less then len(grid);
    x_start : int x coordinate of starting point of BFS should be non negative, less then len(grid[0]);
    num : int number which would be used to enumerate region connected to starting point
  
  Returns
    int : success flag. If BFS was aborted because start point is already visited - 0.
          If BFS finished successfully - 1;
  """
  if enumeration[y_start][x_start] != 0:
    return 0
  y_size = len(grid)
  x_size = len(grid[0])
  visited = [[False for i in range(x_size)] for j in range(y_size)]
  q = deque()
  q.append((y_start, x_start))
  visited[y_start][x_start] = True
  enumeration[y_start][x_start] = num
  while len(q) > 0:
    dot = q.popleft()

    ## Chech if we can up
    if dot[0] > 0 and grid[dot[0] - 1][dot[1]] > 0 and not visited[dot[0] - 1][dot[1]]:
      q.append((dot[0] - 1, dot[1]))
      visited[dot[0] - 1][dot[1]] = True
      enumeration[dot[0] - 1][dot[1]] = num

    ## Chech if we can left
    if dot[1] > 0 and grid[dot[0]][dot[1] - 1] > 0 and not visited[dot[0]][dot[1] - 1]:
      q.append((dot[0], dot[1] - 1))
      visited[dot[0]][dot[1] - 1] = True
      enumeration[dot[0]][dot[1] - 1] = num
    
    ## Chech if we can go down
    if dot[0] < y_size - 1 and grid[dot[0] + 1][dot[1]] > 0 and not visited[dot[0] + 1][dot[1]]:
      q.append((dot[0] + 1, dot[1]))
      visited[dot[0] + 1][dot[1]] = True
      enumeration[dot[0] + 1][dot[1]] = num
    
    ## Chech if we can go right
    if dot[1] < x_size - 1 and grid[dot[0]][dot[1] + 1] > 0 and not visited[dot[0]][dot[1] + 1]:
      q.append((dot[0], dot[1] + 1))
      visited[dot[0]][dot[1] + 1] = True
      enumeration[dot[0]][dot[1] + 1] = num

    ## Chech if we can go left, up
    if (dot[0] > 0 and
        dot[1] > 0 and
        grid[dot[0] - 1][dot[1] - 1] > 0 and
        not visited[dot[0] - 1][dot[1] - 1]):
      q.append((dot[0] - 1, dot[1] - 1))
      visited[dot[0] - 1][dot[1] - 1] = True
      enumeration[dot[0] - 1][dot[1] - 1] = num

    ## Chech if we can go left, down
    if (dot[0] < y_size - 1 and
        dot[1] > 0 and
        grid[dot[0] + 1][dot[1] - 1] > 0 and
        not visited[dot[0] + 1][dot[1] - 1]):
      q.append((dot[0] + 1, dot[1] - 1))
      visited[dot[0] + 1][dot[1] - 1] = True
      enumeration[dot[0] + 1][dot[1] - 1] = num

    ## Chech if we can go right, up
    if (dot[0] > 0 and
        dot[1] < x_size - 1 and
        grid[dot[0] - 1][dot[1] + 1] > 0 and
        not visited[dot[0] - 1][dot[1] + 1]):
      q.append((dot[0] - 1, dot[1] + 1))
      visited[dot[0] - 1][dot[1] + 1] = True
      enumeration[dot[0] - 1][dot[1] + 1] = num

    ## Chech if we can go right, down
    if (dot[0] < y_size - 1 and
        dot[1] < x_size - 1 and
        grid[dot[0] + 1][dot[1] + 1] > 0 and
        not visited[dot[0] + 1][dot[1] + 1]):
      q.append((dot[0] + 1, dot[1] + 1))
      visited[dot[0] + 1][dot[1] + 1] = True
      enumeration[dot[0] + 1][dot[1] + 1] = num
      
  return 1
    
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
from PIL import Image, ImageDraw
import numpy as np

width, height = 300,300
def draw_config(E):
  image = Image.new('RGB', (width, height), (255,255,255))
  image1 = ImageDraw.Draw(image)
  line = [50,250,250,250]
  if E[0] < 0.006:
    line[0] = 50+(1-(E[0]/0.006))*100
  image1.line(line, fill='black', width=20)
  line = [250,250,250,50]
  if E[4] < 0.01:
    line[1] = 250-(1-(E[4]/0.01))*100
  image1.line(line, fill='black', width=20)
  line = [250,50,50,50]
  if E[7] < 0.01:
    line[0] = 250-(1-(E[7]/0.01))*100
  image1.line(line, fill='black', width=20)
  line = [50,50,50,250]
  if E[10] < 0.01:
    line[1] = 50+(1-(E[10]/0.01))*100
  image1.line(line, fill='black', width=20)
  r = np.sqrt(E[0]+E[1]+E[2]+E[3])/0.125*10
  image1.ellipse([250-r,250-r,250+r,250+r], fill='black')
  r = np.sqrt(E[4]+E[5]+E[6])/0.1*10
  image1.ellipse([250-r,50-r,250+r,50+r], fill='black')
  r = np.sqrt(E[7]+E[8]+E[9])/0.1*10
  image1.ellipse([50-r,50-r,50+r,50+r], fill='black')
  r = np.sqrt(E[10]+E[11]+E[12])/0.1*10
  image1.ellipse([50-r,250-r,50+r,250+r], fill='black')
  return image

colors=[(255,0,0),(255,255,0),(0,255,0),(0,255,255),(0,0,255),
        (255,0,255),(127,0,0),(127,127,0)]
'''
deps = \
[[1,0,0,0,0,0,0,0,0,0,1,1,1],
 [1,1,1,0,0,0,0,0,0,0,1,1,1],
 [0,1,1,1,1,0,0,0,0,0,0,0,0],
 [0,0,0,1,1,1,0,0,0,0,0,0,0],
 [0,0,0,0,1,1,1,1,0,0,0,0,0],
 [0,0,0,0,0,0,1,1,1,0,0,0,0],
 [0,0,0,0,0,0,0,1,1,1,1,0,0],
 [0,0,0,0,0,0,0,0,0,1,1,1,0]]
def score_image(image):
  O = np.zeros(8)
  for x in range(width):
    for y in range(height):
      if sum(image.getpixel((x, y))) < 127*3:
        a = np.arctan2(x - 150, y - 150)
        i = int(4 * a / np.pi + 4) % 8
        O[i]+=1
  return O
'''

def score_image(image):
  global count
  O = []
  for j in range(299,0,-100):
    for i in range(0,300,100):
      if j==199 and i==100:
        continue
      O.append(0)
      for i2 in range(100):
        for j2 in range(100):
          if sum(image.getpixel((i+i2,j-j2))) < 127*3:
            O[-1]+=1
  return np.array(O)

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

O_pos = [(0,100,200,300),(100,200,200,300),(200,300,200,300),
           (200,300,100,200),(200,300,0,100),(100,200,0,100),
           (0,100,0,100),(0,100,100,200)]

def score_image(image):
  O = np.zeros((len(O_pos),))
  idx=0
  for L,R,T,B in O_pos:
    for x in range(L,R):
      for y in range(T,B):
        if sum(image.getpixel((x,y))) < 127*3:
          O[idx]+=1
    idx+=1
  return np.array(O)

E_pos = [0,0,1,2,2,3,4,4,5,6,6,7,0]
deps = np.zeros((8,13),dtype=int)
for idx in range(8):
  for end in range(len(E_pos)):
    if E_pos[end] == idx:
      start = max(0,end-4)
      for x in range(start,end+1):
        deps[idx,x]=1

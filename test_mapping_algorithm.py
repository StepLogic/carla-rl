import pickle
import random

import cv2

from topological_map import TopologicalMap


mapper=TopologicalMap()
with open('map.pickle', 'rb') as handle:
    dataset=pickle.load(handle)
# cv2.imwrite("j.jpg",)
images=dataset["images"]
heading=dataset["heading"]

for image,heading in zip(images,heading):
    mapper.update(image,heading)
goal=random.randint(0,len(images)-1)
print("Goal",goal)
subgoal=mapper.create_navigation_guide(images[0],goal)
# breakpoint()
terminate=False
idx=0
while not terminate:
    obs,terminate=subgoal(images[idx])
    idx+=1
    if idx>len(images)-1:
        break

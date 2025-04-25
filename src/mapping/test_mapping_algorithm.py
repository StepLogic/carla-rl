import pickle
import random

import cv2

from src.mapping.topological_map import TopologicalMap


mapper=TopologicalMap()
with open('map.pickle', 'rb') as handle:
    dataset=pickle.load(handle)
features=dataset["features"]
heading=dataset["heading"]

for image,heading in zip(features,heading):
    mapper.update(image,features)
goal=random.randint(0,len(features)-1)
print("Goal",goal)
subgoal=mapper.create_navigation_guide(features[0],goal)
# breakpoint()
terminate=False
idx=0
while not terminate:
    obs,terminate=subgoal(features[idx])
    idx+=1
    if idx>len(features)-1:
        break

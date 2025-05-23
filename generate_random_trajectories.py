import pickle
import random


with open(f'full_trajectory/aux.pkl', 'rb') as handle:
            dataset=pickle.load(handle)

locations=dataset["location"]
user_defined_trajectories=[]
with open(f'full_traj_aux.pkl', 'wb') as handle:
        for _ in range(5):
            start,end=random.choices(locations,k=2)
            user_defined_trajectories.append([start,end])
        pickle.dump(dict(locations=user_defined_trajectories),handle)

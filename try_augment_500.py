import Augmentor
import os

folders = os.listdir(".")

for folder in folders:
	if folder.startswith('.'):
		continue
	if os.path.isdir(folder):
		p = Augmentor.Pipeline(folder)
		p.rotate(probability=1.0, max_left_rotation=5, max_right_rotation=10)
		p.skew(probability=1.0, magnitude=0.5)
		p.rotate(probability=1.0, max_left_rotation=10, max_right_rotation=10)
		p.shear(probability=1.0, max_shear_left=10, max_shear_right=10)
		p.crop_random(probability=0.8, percentage_area=0.8, randomise_percentage_area=False)
		p.crop_random(probability=0.7, percentage_area=0.6, randomise_percentage_area=False)
		p.flip_random(probability=0.7)
		p.sample(500)

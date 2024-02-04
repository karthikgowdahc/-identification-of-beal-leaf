from ultralytics import YOLO

# Load a model

model = YOLO(r"C:\Users\admin\OneDrive\Desktop\Project\yolov8_model1.pt")  # load a custom model
img = "/home/radometech/Documents/Leaf_django/0011_0002.JPG"
results = model(img)  # predict on an image
fg = results[0].probs 
hh = fg.cpu().detach().numpy()
if hh[0]>hh[1]:
	print('Beal')
	model = YOLO(r"C:\Users\admin\OneDrive\Desktop\Project\yolov8_model2.pt")  # load a custom model

	# Predict with the model
	results = model(img)  # predict on an image
	fg = results[0].probs 
	hh = fg.cpu().detach().numpy()
	hh = hh.tolist()
	index_value = hh.index(max(hh))
	if index_value ==0:
		print('Final')
	elif index_value==1:
		print('Initial')
	else:
		print('Intermediate')
else:
	print('Others')

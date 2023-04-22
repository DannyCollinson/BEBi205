Running HW1.ipynb straight through will download the data, make the marker expression panel, define the model, train it, evaluate it, and generate the confusion matrix.
The file HW1_net.pth contains the state_dict for the model, which stores its weights and configuration. It can be loaded by importing pytorch, instantiating an instance of the model class, then calling model.load_state_dict(torch.load(<PATH for HW1_net.pth>)).
The file HW1.pdf has the deliverables for the homework as uploaded to Canvas.
Calling the predict(X,y) method of the HW1Model class in the HW1_model.py file will return a dictionary mapping segmentation mask ids to celltype codes.
The environment.yml file has the conda environment.

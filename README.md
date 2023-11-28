# SAC_JC

Firsly, please download the dataset and our model checkpoint from https://drive.google.com/drive/folders/1fOJayxoNUXm9nmyCjwnxoW1YA0JAGEE3?usp=sharing.
Then, you can leverage SAC.py to evaluate the performance of SAC-JC on KDEF dataset.  

If you want to run this code and train the models on your own, please run  train_teacher.py, train_irrelevant.py, model_extract.py, model_extract_kd.py, fine-pruning.py and adversarial_training.py to get the source model, irrelevant models and the stolen models firsrt.
Then you can leverage dataset_augmented.py to augment data including JPEG compression.
Finally,  you can leverage SAC.py to evaluate the performance of SAC-JC on KDEF dataset.

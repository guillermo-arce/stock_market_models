
Welcome to the README, please find below some indications.

Directory structure explained:

- data: All the raw datasets for the different models. They are picked by "data_processing.py" in order to be pre-processed.

- pytorch: Everything related with the pytorch models (except the raw dataset that is in "data" directory)

---- trained_models: Contains the already trained models, ready to be loaded in "torch_model.py" to make predictions.

---- torch_model.py: The script for creating models, training/loading them and make predictions.

- tensorflow: Everything related with the tensorflow models (except the raw dataset that is in "data" directory)

---- trained_models: Contains the already trained models, ready to be loaded in "tf_model.py" to make predictions.

---- tf_model.py: The script for creating models, training/loading them and make predictions.

- data_processing.py: Script for pre-processing raw data. It is needed to pre-process data before making predictions 
			with the specific model scripts ("tf_model.py" or "torch_model.py")

- ta_functions.py: Auxiliary script for "data_preprocessing.py". It helps with the technical indicators addition.


HOW-TO:

1º Open "data_preprocessing.py" (Spyder IDE is recommended).

2º Execute "data_preprocessing.py" with the desired parameters, just following the script indications.

3º At this point, we can find the pre-processed data and the scaler in "tensorflow" or "pytorch" according to what we have chosen in the "data_preprocessing.py" script.

4º Open "tf_model.py" or "torch_model.py", depending on which data you have pre-processed (Spyder IDE is recommended).

5º Execute it, consistently with parameters specified in "data_preprocessing.py".
	 Again, following the scripts indications will be helpful.

6º Repeat as desired with the different possibilities: TensorFlow with 100/1 or 100/10 models, PyTorch with sept/oct model, PyTorch with oct/nov model and PyTorch with nov/dec model.

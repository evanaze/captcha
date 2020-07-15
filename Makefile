.PHONY: data train eval clean

data:
	# downloads the data from s3 and runs data processing
	aws s3 sync s3://captcha-imgs/ data/raw
	python -m src.data.make_data

train:
	# trains on the full test data with the default settings
	python -m src.models.train --full-model

eval:
	# evaluates OpenCV and Deep Learning Methods
	python -m src.models.eval --square --dcnn

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
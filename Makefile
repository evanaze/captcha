.PHONY: data train eval

data:
	## Download Data from S3
	aws s3 sync s3://captcha-imgs/ data/raw
	python -m src.data.make_data

train:
	python -m src.run -r --save-model

eval:
	python -m src.models.eval --square --dcnn
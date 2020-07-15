.PHONY: data train eval clean

data:
	aws s3 sync s3://captcha-imgs/ data/raw
	python -m src.data.make_data

train:
	python -m src.models.train --save-model

eval:
	python -m src.models.eval --square --dcnn

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
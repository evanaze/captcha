.PHONY: data

data:
	python -m src.data.make_data

train:
	python -m src.run -r 


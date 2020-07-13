.PHONY: data train sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

BUCKET = captcha-imgs
PROFILE = {{ cookiecutter.aws_profile }}

#################################################################################
# COMMANDS                                                                      #
#################################################################################

data:
## Download Data from S3
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/raw
else
	aws s3 sync s3://$(BUCKET)/data/ data/raw --profile $(PROFILE)
endif
	python -m src.data.make_data

train:
	python -m src.run -r 


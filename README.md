# captcha
## Usage
I suggest running scripts from the main directory like modules, as in `python -m src.train`
### Evaluating with the model
I included an evaluate script `src/eval.py`. If you'd like, you can configure the `load` method to take in your image that you would like to run a prediction on. I would suggest using Pillow or `scikit-image` to load the image.

### Retraining the model
If they update the captcha rules or you feel the need to retrain the model, here are the steps to update the model:
1. Create a directory under the main directory called `input`
2. Inside of `input` create a directory called `captcha_examples` with all of your images.
3. Run `src/label.py` to create a csv that contains the data info. The csv should look like:  

| Index | Filename | Target |
|-------|----------|--------|
| 0     | a.png    | 1      |
| 1     | b.png    | 9      |
| 2     | c.png    | 3      | 

4. Run `create_folds.py` to split the data into training, validation, and test
5. Run `train.py` to retrain the model.  

Finally, the model lives in the models directory.

## To-Do
* Experiment more with OpenCV on processed data, try to get cleaner squares
* Ask for more data
* K-Fold CV

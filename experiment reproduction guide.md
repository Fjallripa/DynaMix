# Guide to reproducing the DynaMix Uncertainty experiments

**Contents**
- [experiments](#experiments)
	- [Reproducing experiments / rerunning code](#reproducing-experiments-and-rerunning-code)
- [datasets](#datasets)
- [codebase changes](#codebase-changes)
	- [`forecast_compact()`](#codebase-changes), a 20% faster forecasting implementation

### Experiments
All my experiments were conducted in three Jupyter notebooks: see [`notebooks/Uncertainty quantification ...`](notebooks).
The third - [Uncertainty quantification III - correlation experiments.ipynb](notebooks/Uncertainty%20quantification%20III%20-%20correlation%20experiments.ipynb) - is the most important one. It contains all the final results mentioned in my [report](DynaMix%20project%20report.pdf).

#### Reproducing experiments and rerunning code
**Important:** The headings and subheadings in these notebooks are in **reverse chronological order**, meaning the newest experiments are at the top of the document. The cells inside subheading can be executed in normal chronological order.

When trying to **reproduce** one experiment or just run some of the code, **start with** executing the "Load model and data" section at the top, then jump to whichever experiment section you want. Sometimes there's code at the top of a section that needs to be executed before jumping to the subsection of your choice.

### Datasets
Due to GitHub upload size limitations, I unfortunately **couldn't save the prediction datasets** I created. If a dataset is missing, the code to recreate it can be found at the relevant sections of the notebook and, if needed, their specific settings are detailed as JSON files in the [`noteboooks/predictions/`](notebooks/predictions) folder.

Note: The **'short-term forecasting dataset'** (responsible for all the main results) wasn't saved and took over 6 hours to create on my laptop.

The **simulated datasets, I could save all** under [`notebooks/test_data/`](notebooks/test_data).


### Codebase changes
During my experiments, I needed to tweak the `Dynamix` and `DynamixForecaster` classes multiple times. I tried to keep everything back-compatible but am not sure if that worked.

For [`dynamix.py`](src/model/dynamix.py), the most important change is addition of keyword argument `attention_noise=True` to `GatingNetwork.forward()` and methods calling that one. This allows for toggling of attention noise as well as giving an explicit noise tensor as input for reproducability.

#### `forecast_compact()`
For [`forecaster.py`](src/model/forecaster.py), I added multiple `forecast...()` methods to `DynamixForecaster`. Most are only of interest now for exactly reproducing some of my experiments. Only the last one, **`forecast_compact()`** should be of remaining interest. This implementation **runs 20% faster** than the original one. 

I also did a lot of code refactoring on this one, making it a lot more compact. In particular I collect the forecasting loop's code that was spread out all over the `DynaMix` class and put it into one well-structured for-loop that makes the **prediction step process** of DynaMix **a lot more glancable**. However, I also narrowed down `forecast_compact()`'s preprocessing to the default options, so while more readable, it isn't a drop-in replacement for the standard `forecast_method()`.

The **main speed-up** is most likely **due to running the 10 expert ALRNN's in parallel** instead of in a for-loop as the original did, but I haven't verified this yet. In any case, this seems like a sensible change to implement in the main `forecast()` method as well.

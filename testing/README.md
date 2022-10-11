# Testing
Testing the models works in three phases. First, the model runs over all the test data and its output gets saved (Pretest).
Second, the output of the model is compared to that of the ground truth panel (Compare).
Third, the output of the models can also be compared visually (Plotting).

## Pretest
To run the models and the ground truth over the test set, please run the following commands:
```
python pretest_GroundTruth.py -data_loc="target_folder"
python pretest_U-Flow.py -data_loc="target_folder"
python pretest_U-Net.py -data_loc="target_folder"
```

## Compare
To compare the results of the output of the model against the panel of scorers run:
```
python Compare.py -name="U-Flow"
python Compare.py -name="U-Net"
```

## Plotting
The results can also be plotted visually using the following commands:
```
python plot_Results.py -name="ground_truth"
python plot_Results.py -name="U-Flow"
python plot_Results.py -name="U-Net"
```
The resulting figures can be found in the 'figures' subfolder.

## Note on Stochasticity
As all models under consideration have some degree of stochastic behaviour, it is expected that replicating the results on different machines will lead to slightly different results. 
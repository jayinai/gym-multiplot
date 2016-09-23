# gym-multiplot
Visualizing and Comparing Multiple OpenAI Gym Experiments

## Ever wanted to visualize and compare multiple experimental runs from OpenAI?
Me too.

## How do I use this?
1. Start with the notebook, see if it fits your needs.
2. Copy over some experimenal results worth comparing, there are currently a few example folders here so that you can get a sense of the structure and funcionality.
3. Revel in the beautiful aggregate statistics.

## More details
Currently, gym-multiplot displays average and standard deviation for episode rewards and lengths. 
Figures show mean in black and each run in a different slightly transparent color, linked to the legend. Standard deviation is the shaded gray region around the mean.

## But korymath, it's not working! 
First, this was a proof of concept hackup. 
Second, drop an issue and let's figure out how to make this work more robustly together.

## What's next?
There are lots more statistics to capture from OpenAI gym, and this just shows a single sample... 

![length example](https://github.com/korymath/gym-multiplot/blob/master/images/example-length.png?raw=true)

![reward example](https://raw.githubusercontent.com/korymath/gym-multiplot/master/images/example-reward.png)

## Thanks
[JKCooper2](https://github.com/JKCooper2) for the quick help early on.

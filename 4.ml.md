# ML
An agent is learning if it improves its performance on future tasks after making observations
about the world.

There are three types of feedback that determine the three main types of learning:
- unsupervised learning: the agent learns patterns in the input even though no explicit feedback is supplied.
- reinforcement learning: the agent learns from a series of reinforcements—rewards or punishments.
- supervised learning: the agent observes some example input–output pairs and learns a function that maps from input to output.

<img src="https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2013/11/Supervised-Learning-Algorithms.png">
<img src="https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2013/11/Supervised-Learning-Algorithms.png">


## Unsupervised learning
### KNN (K- Nearest Neighbors)
### K-Means

### Mini project: Clustring posts

```python
import os
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1)
posts = [open(os.path.join("./posts", f)).read() for f in os.listdir("./posts")]
X_train = vectorizer.fit_transform(posts)

new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post])
print(new_post_vec)
```

## Supervised learning

<iframe width="560" height="315" src="https://www.youtube.com/embed/ACmydtFDTGs" frameborder="0" gesture="media" allow="encrypted-media" allowfullscreen></iframe>

### LEARNING DECISION TREES
Example of data:

<img src="https://preview.ibb.co/n1Nb9w/Screen_Shot_2018_01_04_at_13_50_11.png" alt="Screen_Shot_2018_01_04_at_13_50_11" border="0">

The result decision tree:

<img src="http://www.cs.bham.ac.uk/~mmk/Teaching/AI/figures/dectree-orig.jpg">


### Support vector machine


### Regression
x = features
y = goal

```python
fp = sp.polyfit(x, y, 5)
f  = sp.poly1d(fp)
```

### ARTIFICIAL NEURAL NETWORKS


## reinforcement learning
https://github.com/openai/universe

Genetic ML
https://www.youtube.com/watch?v=aeWmdojEJf0

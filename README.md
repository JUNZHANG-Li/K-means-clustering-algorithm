# K means clustering algorithm

This is a K means clustering algorithm inplemented in K means ++

To implement this algorithm, I find the first group center in random at first, and find rest of the group center based on the distance to other exist centers. Then applied gradient descent to update the position of centers until their locations are converged. Finally classified each point on the graph to their closest group (center of group).

Here is the result of this algorithm:
<img width="752" alt="common result of K means ++ algorithm" src="https://user-images.githubusercontent.com/91993785/198905902-1e8ee34e-7e10-4a8a-8385-a2a315fb45e9.png">

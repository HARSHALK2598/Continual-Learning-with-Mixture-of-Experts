# Continual-Learning-with-Mixture-of-Experts
Continual LearningWith Mixture of Experts using subnetwork selection and masking

Existing research in the task-agnostic continual learning setting use loss function
value as an indicator for the onset of a new task and hence creating a new instance
of an expert for the incoming new task. Although this prevents catastrophic
forgetting, these methods are highly inefficient and prevent knowledge transfer.
The goal of the proposed method is to create diverse models over time, where
each model focuses on training tasks with similar parameter spaces (benefitting
from knowledge transfer and catastrophic remembering) along with alleviating
catastrophic forgetting. The gradients of the loss with respect to the parameters
after convergence are used to identify the useful subnetwork for the task at hand,
and freeze them for training the new task on the remaining subnetwork, enabling
knowledge transfer and alleviating catastrophic forgetting, creating an efficient
system as a whole.

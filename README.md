This project demonstrates image reconstruction by solving optimization problems with missing pixel values. Specifically, it uses Accelerated Gradient Descent (AGD) and Stochastic Gradient Descent (SGD) methods to recover missing pixels in an image.

Overview
Given an image with missing pixels, the goal of this project is to reconstruct the original image using optimization techniques. The image is initially corrupted by randomly removing pixels. Then, two optimization methods, Accelerated Gradient Descent and Stochastic Gradient Descent, are used to recover the image data by minimizing the nuclear norm of the matrix representing the image.

The performance of both methods is tracked by monitoring the stopping criterion and the execution time. The final output includes the reconstructed image, and the convergence behavior of the methods is displayed via plots.

Techniques Used
Accelerated Gradient Descent (AGD):

Implements the AGD method to iteratively update the pixel values while minimizing the nuclear norm subject to the known pixels.
Stochastic Gradient Descent (SGD):

Implements the SGD method by randomly selecting a subset of the available pixels (batch) to perform gradient updates.
Nuclear Norm Minimization:

The reconstruction problem is modeled as a convex optimization problem with a nuclear norm constraint, ensuring that the recovered image has the smallest possible singular values.

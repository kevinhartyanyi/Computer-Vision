import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import random
from point2d import *
import sys


def plot_image(axis,img):
    axis.imshow(img, cmap='gray')
    axis.set_xticks([])
    axis.set_yticks([])

picture = '0097.jpg'

original = cv.imread(picture)
edges = cv.Canny(original,100,200)
edges_rgb = cv.cvtColor(edges,cv.COLOR_GRAY2RGB)

edges_binary = np.asarray(edges, dtype=bool)

row,column = np.where(edges_binary == 1)

edge_points = [Point2D(x,y) for x,y in list(zip(column,row))]



"""for p in edge_points:
    edges_rgb = cv.circle(edges_rgb, p.cartesian(), 1, (255,0,0))
plt.imshow(edges_rgb, cmap='gray')
plt.show()"""


def FitLineRANSAC(points, threshold, it_num, image):
    it = 0
    bestInlinerNum = 0
    bestInliners = []
    bestLine = []
    bestPt1, bestPt2 = Point2D(), Point2D()  # Helpers

    while it < it_num:
        print(it)
        img = image.copy()

        # 1. Select a minimal sample i.e. 2 random points
        sample = random.sample(range(0, len(points)), 2)


        # 2. Fit a line to the points

        v = points[sample[1]] - points[sample[0]]
        v.r = 1
        n = Point2D(-v.x, v.y)

        a = n.x
        b = n.y
        c = -a * points[sample[1]].x - b * points[sample[0]].y

        # Get the inliners

        inlinerNumber = 0
        inliners = []
        for p in points:
            distance = abs(a * p.x + b * p.y + c)
            if distance < threshold:
                inlinerNumber += 1
                inliners.append(p)
                # Draw points
                img = cv.circle(img, p.cartesian(), 2, (255,0,0))

        pt1 = Point2D(int(round(-c / a)), 0)
        pt2 = Point2D(int(round(-(c + b * img.shape[0]) / a)), img.shape[0])

        print(pt1.cartesian(), pt2.cartesian())
        if len(inliners) > 20:
            image = cv.line(img, pt1.cartesian(), pt2.cartesian(), (255, 0, 0), 1)

        """# Current line
        img = cv.line(img, pt1.cartesian(), pt2.cartesian(), (255,0,0), 1)

        if bestInlinerNum > 0:
            # Best line
            img = cv.line(img, bestPt1.cartesian(), bestPt2.cartesian(), (0, 255, 0), 1)   

        if inlinerNumber > bestInlinerNum:
            bestInlinerNum = inlinerNumber
            bestInliners = inliners
            bestLine = (a,b,c)

            bestPt1 = pt1
            bestPt2 = pt2"""

        it += 1

    return bestInliners, bestLine, image



bestInliners, bestLine, line_image = FitLineRANSAC(edge_points, 2.0, int(sys.argv[1]), edges_rgb)
plt.imsave(f"{picture.split('.')[0]}_result.png", line_image)

fig, axes = plt.subplots(nrows=1,ncols=3, figsize=(50,50))
plot_image(axes[0], original)
plot_image(axes[1], edges)
plot_image(axes[2], line_image)

plt.show()





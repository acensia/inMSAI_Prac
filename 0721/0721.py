import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

boxes = np.array([[10, 10, 50, 50],
         [100, 90, 150, 150],
         [30, 40, 80, 100],
         [50, 60, 120, 160],
         [20, 30, 70, 90],
         [80, 70, 140, 180]])

def kmeans(boxes, k, num_iters=100):
    num_boxes = boxes.shape[0]
    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    indices = np.argsort(box_areas)
    
    clusters = boxes[indices[-k :]]
    
    prev_clusters = np.zeros_like(clusters)
    
    
    for _ in range(num_iters):
        box_clusters = np.argmin(((boxes[:, None] - clusters[None]) ** 2).sum(axis=2), axis=1)
        print(boxes[:, None])
        print(clusters[None])
        print((boxes[:, None] - clusters[None]))
        print(box_clusters)
        
        for cluster_idx in range(k):
            if np.any(box_clusters == cluster_idx):
                clusters[cluster_idx] = boxes[box_clusters == cluster_idx].mean(axis=0)
                
        if np.all(np.abs(prev_clusters - clusters) < 1e-6):
            break
        
        prev_clusters = clusters.copy()
    return clusters
        
def plot_boxes(boxes, title="Anchors"):
    fig, ax = plt.subplots(1)
    ax.set_title(title)
    
    img_width, img_height = 200, 200
    
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        x_min, x_max = x_min / img_width, x_max/img_width
        y_min, y_max = y_min / img_height, y_max / img_height
        
        w, h = x_max - x_min, y_max - y_min
        rect = patches.Rectangle((x_min, y_min), w, h, linewidth=1, edgecolor='r', facecolor='none')
        
        ax.add_patch(rect)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    plt.show()
    
            
        

        
        
anchors = kmeans(boxes, 5, 1)


print("anc")
print(anchors)

plot_boxes(anchors)
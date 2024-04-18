import numpy as np


def compress_img(img, downsize_by, iters):
    """
    Downsizes image by reducing its colors.
    Colors are computed by clustering, K-mean algorythm
    :param img: image as matrix of colors (array of nd vectors)
    :param downsize_by: 0.0 for only one color in final image, 1.0 for no compress at all
    :param iters: number of centroid sets to choose from
    :return: compressed image
    """

    img = np.array(img)
    init_colors = _get_colors(img)
    k = np.floor(float(len(init_colors)) * downsize_by)
    if k == 0:
        k = 1
    best_centroids = _get_optimal_centroids(img, k, iters)

    return recolor_img(img, best_centroids)


def compress_img_by_k(img, k, iters):
    """
    Downsizes image by reducing its colors.
    Colors are computed by clustering, K-mean algorythm
    :param k: colors number
    :param img: image as matrix of colors (array of nd vectors)
    :param iters: number of centroid sets to choose from
    :return: compressed image
    """

    img = np.array(img)
    init_colors = _get_colors(img)
    if k <= 0:
        k = 1
    best_centroids = _get_optimal_centroids(img, init_colors, k, iters)

    return recolor_img(img, best_centroids)


def recolor_img(img, colors):
    new_img = img
    cid = _get_cid(img, colors)
    m = img.shape[0]
    for i in range(m):
        new_img[i] = colors[int(cid[i])]
    return new_img


def _get_optimal_centroids(x, start_points, k, iters):
    """
    :param x: data set
    :param k: centroids number
    :param iters: num of centroid sets to choose from
    :return: the best centroid set of computed
    """

    all_centroids = []
    all_costs = np.zeros((iters,))
    for i in range(iters):
        centroids = _fit_centroids(x, start_points, k)
        cid = _get_cid(x, centroids)
        all_centroids.append(centroids)
        all_costs[i] = _compute_cost(x, cid, centroids)
    best_id = np.argmin(all_costs)
    return all_centroids[best_id]


def _fit_centroids(x, start_points, k):
    """
    :param x: data set
    :param k: centroids number
    :return: computed centroids
    """

    min_delta = 1
    centroids = _get_random_centroids(start_points, k)
    prev_cost = -min_delta
    cost = min_delta
    while np.abs(cost - prev_cost) > min_delta:
        cid = _get_cid(x, centroids)
        centroids = _compute_centroids(x, cid, k)
        prev_cost = cost
        cost = _compute_cost(x, cid, centroids)
    return centroids


def _get_colors(img):
    """
    :param img: image
    :return: all unique colors of the image
    """

    colors_map = dict()
    m = img.shape[0]
    for i in range(m):
        color = img[i].tolist()
        color_tuple = tuple(color)
        colors_map[color_tuple] = 1
    colors = []
    color_tuples = list(colors_map.keys())
    m = len(color_tuples)
    for i in range(m):
        colors.append(list(color_tuples[i]))
    return colors


def _get_random_centroids(v, k):
    """
    :param v: set of colors to choose
    :param k: number of total centroids
    :return: random centroids
    """
    ids = list(range(np.array(v).shape[0]))
    rand = np.random.choice(ids, size=k, replace=False)
    centroids = []
    for i in range(k):
        centroids.append(v[rand[i]])
    return np.array(centroids)


def _get_cid(x, centroids):
    """
    :param x: data set
    :param centroids: cluster centroids
    :return: id set of the closest centroid for each train example
    """

    m = x.shape[0]
    k = centroids.shape[0]
    cid = np.zeros((m,))
    for i in range(m):
        distances = np.zeros((k,))
        for j in range(k):
            distances[j] = np.linalg.norm(x[i] - centroids[j])
        cid[i] = np.argmin(distances)
    return cid


def _compute_centroids(x, cid, k):
    """
    :param x: data set
    :param cid: id set of the closest centroid for each train example
    :param k: number of clusters
    :return: centers of each cluster
    """

    centroids = []
    for i in range(k):
        centroids.append(np.mean(x[cid == i], axis=0))
    return np.array(centroids)


def _compute_cost(x, cid, centroids):
    """
    :param x: data set
    :param cid: id set of the closest centroid for each train example
    :param centroids: cluster centroids
    :return: cost function value ( average of ||x_i - centroid_i||^2 )
    """

    m = x.shape[0]
    cost = 0
    for i in range(m):
        v = x[i] - centroids[int(cid[i])]
        cost += np.linalg.norm(v)**2
    cost /= m
    return cost

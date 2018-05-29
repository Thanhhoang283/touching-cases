import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import cv2

def show_img(img, winname="TEST"):
    plt.figure()
    plt.imshow(img)
    plt.title(winname)
    plt.show()

def convert_binary_to_normal_im(binary):
    gray_img = ((1-binary)*255).astype('uint8')
    return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

# get neighbors d index
def neighbors(shape):
    dim = len(shape)
    block = np.ones([3] * dim)
    block[tuple([1] * dim)] = 0
    idx = np.where(block > 0)
    idx = np.array(idx, dtype=np.uint8).T
    idx = np.array(idx - [1] * dim)
    acc = np.cumprod((1,) + shape[::-1][:-1])
    return np.dot(idx, acc[::-1])

# mark the array use (0, 1, 2)
def mark(img):
    nbs = neighbors(img.shape)
    img = img.ravel()
    for p in range(len(img)):
        if img[p] == 0: continue
        s = 0
        for dp in nbs:
            if img[p + dp] != 0: s += 1
        if s == 2:
            img[p] = 1
        else:
            img[p] = 2


# trans index to r, c...
def idx2rc(idx, acc):
    rst = np.zeros((len(idx), len(acc)), dtype=np.int16)
    for i in range(len(idx)):
        for j in range(len(acc)):
            rst[i, j] = idx[i] // acc[j]
            idx[i] -= rst[i, j] * acc[j]
    rst -= 1
    return rst


# fill a node (may be two or more points)
def fill(img, p, num, nbs, acc, buf):
    back = img[p]
    img[p] = num
    buf[0] = p
    cur = 0
    s = 1

    while True:
        p = buf[cur]
        for dp in nbs:
            cp = p + dp
            if img[cp] == back:
                img[cp] = num
                buf[s] = cp
                s += 1
        cur += 1
        if cur == s: break
    return idx2rc(buf[:s], acc)


# trace the edge and use a buffer, then buf.copy, if use [] numba not works
def trace(img, p, nbs, acc, buf):
    c1 = 0;
    c2 = 0;
    newp = 0
    cur = 0

    while True:
        buf[cur] = p
        img[p] = 0
        cur += 1
        for dp in nbs:
            cp = p + dp
            if img[cp] >= 10:
                if c1 == 0:
                    c1 = img[cp]
                else:
                    c2 = img[cp]
            if img[cp] == 1:
                newp = cp
        p = newp
        if c2 != 0: break
    return (c1 - 10, c2 - 10, idx2rc(buf[:cur], acc))


# parse the image then get the nodes and edges
def parse_struc(img):
    nbs = neighbors(img.shape)
    # a list has length that is euqual to img.shape
    acc = np.cumprod((1,) + img.shape[::-1][:-1])[::-1]
    # print(acc)
    img = img.ravel()
    pts = np.array(np.where(img == 2))[0]
    buf = np.zeros(len(pts), dtype=np.int64) # ????

    num = 10
    nodes = []
    for p in pts:
        if img[p] == 2:
            nds = fill(img, p, num, nbs, acc, buf)
            num += 1
            nodes.append(nds)
            # for i in nds:
            #     nodes.append(i)
    edges = []
    for p in pts:
        for dp in nbs:
            # edge = trace(img, p + dp, nbs, acc, buf)
            # edges.append(edge)
            if img[p + dp] == 1:
                edge = trace(img, p + dp, nbs, acc, buf)
                # print(edge)
                edges.append(edge)
    # print(len(edges))
    return nodes, edges


# use nodes and edges build a networkx graph
def build_graph(nodes, edges, multi=False):
    graph = nx.MultiGraph() if multi else nx.Graph()
    for i in range(len(nodes)):
        graph.add_node(i, pts=nodes[i], o=nodes[i].mean(axis=0))
    for s, e, pts in edges:
        l = np.linalg.norm(pts[1:] - pts[:-1], axis=1).sum()
        graph.add_edge(s, e, pts=pts, weight=l)
    return graph

def buffer(ske):
    buf = np.zeros(tuple(np.array(ske.shape) + 2), dtype=np.uint16)
    buf[tuple([slice(1, -1)] * buf.ndim)] = ske
    return buf


def build_sknw(ske, multi=False):
    buf = buffer(ske)
    mark(buf)
    # show_img(buf)
    nodes, edges = parse_struc(buf)
    # print(len(nodes))
    return build_graph(nodes, edges, multi)



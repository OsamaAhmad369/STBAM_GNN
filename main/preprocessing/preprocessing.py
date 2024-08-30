import os
import argparse
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import skimage as io
from skimage.segmentation import slic, mark_boundaries
from skimage.color import lab2rgb
from skimage.future import graph
from sklearn.model_selection import train_test_split
import pickle
import h5py
import torch
from torch_geometric.data import Data, Batch
import torchvision.models as models
from torch.nn import Sequential
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj,to_dense_batch,dense_to_sparse

class preprocessing:
    def __init__(self,nodes=64,compactness=10,model_path=None):
        self.nodes=int(nodes)
        self.row=224
        self.col=224
        self.compactness=compactness
        self.frame_len=3
        self.model_path=model_path
        vgg16=models.vgg16(weights='IMAGENET1K_V1')
        self.new_model=Sequential(*list(vgg16.features.children())[:2])
        self.new_model.eval()
        
    # Preprocess the dataset
    def preprocess_image(self,image):
        superpixels_mask = slic(image, n_segments=self.nodes, compactness=self.compactness)  
        output_features=self.VGG16(torch.tensor(image,dtype=torch.float32).permute(2,0,1))
        node_features = self.compute_superpixel_features(image, superpixels_mask, output_features)
        adjacency_matrix = self.compute_adjacency_matrix(image, superpixels_mask)
        return torch.tensor(node_features, dtype=torch.float), torch.tensor(adjacency_matrix, dtype=torch.float)

    def compute_superpixel_features(self,image, segments,output_features):
        num_superpixels = int(np.max(segments))
        superpixel_features = np.zeros((num_superpixels, 67))
        output_features=output_features.permute(2,1,0)
        _features=np.array(output_features.tolist())
        image=np.concatenate((image,_features),axis=2)
        for i in range(num_superpixels-1):
            i+=1
            mask = segments == i     
            superpixel_pixels = image[mask.reshape(image.shape[:-1])]  # Reshape the mask to match image shape
            avg_color = np.max(superpixel_pixels, axis=0)
            superpixel_features[i] = avg_color
        return superpixel_features

    def compute_adjacency_matrix(self,image, superpixels):
        # Create a region adjacency graph based on the mean color of each superpixel
        rag = graph.rag_mean_color(image, superpixels)
        # Convert RAG to adjacency matrix
        num_superpixels = int(np.max(superpixels)) 
        adjacency_matrix = np.zeros((num_superpixels, num_superpixels))
        for edge in rag.edges():
            i, j = edge
            i-=1
            j-=1
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1
        # edge_index = np.stack(np.where(adjacency_matrix == 1))
        return adjacency_matrix

    def VGG16(self,image):
        output_features=self.new_model(image)
        output_features=output_features.detach()
        return output_features
    
    def split_data(self,data,labels): 
        train_data, test_data, train_labels, test_labels=train_test_split(data,labels,test_size=0.15,random_state=42)
        train_data, val_data, train_labels, val_labels=train_test_split(train_data,train_labels,test_size=0.15,random_state=42)
        return train_data, train_labels, val_data, val_labels, test_data, test_labels
    
    def save_data(self, data_list, data_type):
        with open(f"{self.model_path}/{data_type}_c2d2_BA.pkl", "wb") as f:
            pickle.dump(data_list, f)
    def run(self):
        file_path = os.path.join(self.model_path, "dataset-001.h5")
        file = h5py.File(file_path)
        labels = np.array(file['data_y'])
        data = np.array(file['data_x'])

        train_data, train_labels, val_data, val_labels, test_data, test_labels = self.split_data(data, labels)

        datasets = {'train': (train_data, train_labels), 'val': (val_data, val_labels), 'test': (test_data, test_labels)}
       
        for index, (data_type, (_data, labels)) in enumerate(datasets.items()):
            data_list = []
            for i in range(len(_data)):
                label = labels[i]
                _images = _data[i] / 255.
                adj={}
                x={}
                BA_size=self.nodes*self.frame_len
                block_adj=torch.zeros((BA_size,BA_size))
                prev_shape=0
                for j in range(self.frame_len):
                    image = cv2.resize(_images[j], (self.row, self.col))
                    _x, adj[j] = self.preprocess_image(image)
                    super_pixel_nodes=adj[j].shape[0]
                    if  super_pixel_nodes <self.nodes:
                        _adj=torch.cat((adj[j],torch.zeros((self.nodes-super_pixel_nodes,super_pixel_nodes))),axis=0)
                        _adj=torch.cat((_adj,torch.zeros((self.nodes,self.nodes-super_pixel_nodes))),axis=1)
                        x[j]=torch.cat((_x,torch.zeros((self.nodes-super_pixel_nodes,67))),axis=0)
                    elif super_pixel_nodes>self.nodes:
                        _adj=adj[j][:-1,:]
                        _adj=_adj[:,:-1]
                        x[j]=_x[:-1,:]
                    else:
                        x[j]=_x
                        _adj=adj[j]
                    new_shape=self.nodes+prev_shape
                    block_adj[prev_shape:new_shape,prev_shape:new_shape]=_adj
                    prev_shape=new_shape
                    
                _edge_attr=[adj[0][0].shape[0], adj[1][0].shape[0],adj[2][0].shape[0]] #edge attribute contains the information regarding the dynamic number of nodes 
                _edge_index,_=dense_to_sparse(block_adj)
                
                xnew=torch.cat([x[0],x[1],x[2]],dim=0)
                data = Data(x=xnew,edge_index=_edge_index,edge_attr=_edge_attr,y=[label])
                data_list.append([data])
        
            self.save_data(data_list, data_type)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument("--nodes", type=int, default=64, help="Number of nodes for superpixels")
    parser.add_argument("--compactness", type=float, default=10.0, help="Compactness for superpixel segmentation")
    parser.add_argument("--path", type=str, default='data/', help="Path to the dataset directory")
    
    args = parser.parse_args()

    processor = preprocessing(nodes=args.nodes, compactness=args.compactness, model_path=args.path)
    processor.run()
    

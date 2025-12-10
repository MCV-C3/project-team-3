from typing import List, Tuple, Type, Literal
from PIL import Image
import numpy as np
from bovw_kunal import BOVW_kunal
import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def extract_bovw_histograms_kunal(bovw: Type[BOVW_kunal], descriptors: Literal["N", "T", "d"]):
    return np.array([bovw._compute_codebook_descriptor_kunal(descriptors=descriptor, kmeans=bovw.codebook_algo) for descriptor in descriptors])

def extract_descriptors(dataset: List[Tuple[np.ndarray, int]],
                        bovw: Type[BOVW_kunal]) -> List[Tuple[np.ndarray, int]]:
    descriptors_dataset = []
    
    for idx in tqdm.tqdm(range(len(dataset)), desc="Phase [Training]: Extracting the descriptors"):
        
        image, label = dataset[idx]
        _, descriptors = bovw._extract_features(image=np.array(image))
        
        if descriptors is not None:
            descriptors_dataset.append((descriptors, label))
    return descriptors_dataset    

def train_descriptors(descriptors_dataset: List[Tuple[np.ndarray, int]],
          bovw: Type[BOVW_kunal]):
    all_descriptors = []
    all_labels = []
    for idx in range(len(descriptors_dataset)):
        descriptors, label = descriptors_dataset[idx]
        all_descriptors.append(descriptors)
        all_labels.append(label)
    
    print("Fitting the codebook")
    kmeans, cluster_centers = bovw._update_fit_codebook(descriptors=all_descriptors)

    print("Computing the bovw histograms")
    bovw_histograms = extract_bovw_histograms_kunal(descriptors=all_descriptors, bovw=bovw) 
    
    print("Fitting the classifier")
    classifier = LogisticRegression(class_weight="balanced").fit(bovw_histograms, all_labels)

    print("Accuracy on Phase[Train]:", accuracy_score(y_true=all_labels, y_pred=classifier.predict(bovw_histograms)))
    
    return bovw, classifier

def test_descriptors(descriptors_dataset: List[Tuple[np.ndarray, int]],
         bovw: Type[BOVW_kunal], 
         classifier:Type[object]):
    
    test_descriptors = []
    descriptors_labels = []
    
    for idx in range(len(descriptors_dataset)):
        descriptors, label = descriptors_dataset[idx]
        test_descriptors.append(descriptors)
        descriptors_labels.append(label)            
    
    print("Computing the bovw histograms")
    bovw_histograms = extract_bovw_histograms_kunal(descriptors=test_descriptors, bovw=bovw)
    
    print("predicting the values")
    y_pred = classifier.predict(bovw_histograms)
    
    acc = accuracy_score(y_true=descriptors_labels, y_pred=y_pred)
    print("Accuracy on Phase[Test]:", acc)

    return acc   

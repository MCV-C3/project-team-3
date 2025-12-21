import cv2
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import os
import glob
from typing import List, Tuple, Union


class BOVW():
    
    def __init__(self, descriptor_type="SIFT", codebook_size: int = 50, 
                 codebook_type: str = "kmeans", detector_kwargs: dict = {}, 
                 codebook_kwargs: dict = {}):
        """
        Initialize Bag of Visual Words model
        
        Args:
            descriptor_type: Type of descriptor ('SIFT', 'AKAZE', 'ORB', 'DENSE_SIFT', 'precomputed')
            codebook_size: Number of visual words in the codebook
            codebook_type: Type of codebook ('kmeans' or 'gmm')
            detector_kwargs: Additional arguments for the detector
            codebook_kwargs: Additional arguments for the codebook algorithm
        """
        
        self.descriptor_type = descriptor_type
        self.codebook_size = codebook_size
        self.codebook_type = codebook_type
        self.kp = None
        
        # Initialize detector only if not using precomputed features
        if descriptor_type != 'precomputed':
            if descriptor_type == 'SIFT':
                self.detector = cv2.SIFT_create(**detector_kwargs)
            elif descriptor_type == 'AKAZE':
                self.detector = cv2.AKAZE_create(**detector_kwargs)
            elif descriptor_type == 'ORB':
                self.detector = cv2.ORB_create(**detector_kwargs)
            elif descriptor_type == 'DENSE_SIFT':
                step_size = detector_kwargs.get('step_size', 8)
                kp_size = detector_kwargs.get('kp_size', step_size)

                detector_kwargs_copy = detector_kwargs.copy()
                detector_kwargs_copy.pop('step_size', None)
                detector_kwargs_copy.pop('kp_size', None)

                self.detector = cv2.SIFT_create(**detector_kwargs_copy)
                self.kp = (lambda image: [
                    cv2.KeyPoint(x, y, kp_size)
                    for y in range(0, image.shape[0], step_size) 
                    for x in range(0, image.shape[1], step_size)
                ])
            else:
                raise ValueError("Descriptor type must be 'SIFT', 'AKAZE', 'DENSE_SIFT', 'ORB', or 'precomputed'")
        else:
            self.detector = None

        # Initialize codebook algorithm
        if codebook_type == "kmeans":
            self.codebook_algo = MiniBatchKMeans(n_clusters=self.codebook_size, 
                                                 batch_size=1024,
                                                 random_state=42,
                                                 **codebook_kwargs)
        elif codebook_type == "gmm":
            self.codebook_algo = GaussianMixture(n_components=self.codebook_size,
                                                 covariance_type='diag',
                                                 random_state=42,
                                                 **codebook_kwargs)
        else:
            raise ValueError("Codebook type must be 'kmeans' or 'gmm'")
        
        self.codebook = None
        self.is_fitted = False
               
    def _extract_features(self, image: np.ndarray) -> Tuple:
        """Extract features from an image using the configured detector"""
        if self.descriptor_type == 'precomputed':
            raise ValueError("Cannot extract features with 'precomputed' descriptor type")
        
        if self.kp is not None:
            # Dense SIFT
            keypoints, descriptors = self.detector.compute(image, self.kp(image))
        else:
            # Standard feature detection and description
            keypoints, descriptors = self.detector.detectAndCompute(image, None)
        
        return keypoints, descriptors
    
    def fit_codebook(self, descriptors_list: Union[List[np.ndarray], np.ndarray], batch_size: int = 10000):
        """
        Fit the codebook on a list of descriptor arrays or a single large array
        
        Args:
            descriptors_list: Either:
                - List of descriptor arrays, each of shape [num_descriptors, descriptor_dim]
                - Single array of shape [total_descriptors, descriptor_dim]
            batch_size: Batch size for mini-batch training
        """
        print(f"Fitting codebook...")
        
        # Handle single array case
        if isinstance(descriptors_list, np.ndarray):
            descriptors_list = [descriptors_list]
        
        if self.codebook_type == "kmeans":
            # Use MiniBatchKMeans for memory efficiency
            total_processed = 0
            for i, descriptors in enumerate(descriptors_list):
                if descriptors is None or len(descriptors) == 0:
                    continue
                
                # Ensure descriptors are float32
                descriptors = descriptors.astype(np.float32)
                
                print(f"  Processing descriptor set {i+1}/{len(descriptors_list)}: {descriptors.shape}")
                
                # Split into mini-batches
                for j in range(0, len(descriptors), batch_size):
                    chunk = descriptors[j : j + batch_size]
                    self.codebook_algo.partial_fit(chunk)
                    total_processed += len(chunk)
                
                if (i + 1) % 10 == 0 or i == len(descriptors_list) - 1:
                    print(f"  Processed {total_processed:,} descriptors so far...")
            
            self.codebook = self.codebook_algo.cluster_centers_
            
        elif self.codebook_type == "gmm":
            # GMM requires all data, so we sample to manage memory
            print("  Sampling descriptors for GMM (memory constraint)...")
            sampled = []
            descriptors_per_image = 500
            
            for descriptors in descriptors_list:
                if descriptors is None or len(descriptors) == 0:
                    continue

                # Sample descriptors
                if len(descriptors) > descriptors_per_image:
                    idx = np.random.choice(len(descriptors), descriptors_per_image, replace=False)
                    sampled.append(descriptors[idx])
                else:
                    sampled.append(descriptors)
            
            sampled = np.vstack(sampled).astype(np.float64)
            print(f"  Total sampled descriptors: {len(sampled):,}")
            
            self.codebook_algo.fit(sampled)
            self.codebook = self.codebook_algo.means_
        
        self.is_fitted = True
        print(f"✓ Codebook fitted with {self.codebook_size} visual words")
        print(f"  Codebook shape: {self.codebook.shape}")
        
    def compute_histogram(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Compute BoVW histogram for a set of descriptors
        
        Args:
            descriptors: Array of descriptors [num_descriptors, descriptor_dim]
            
        Returns:
            Normalized histogram of visual words
        """
        if not self.is_fitted:
            raise ValueError("Codebook must be fitted before computing histograms")
        
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.codebook_size)
        
        descriptors = descriptors.astype(np.float32)
        
        if self.codebook_type == "kmeans":
            # Predict visual word for each descriptor
            visual_words = self.codebook_algo.predict(descriptors)
            
            # Create histogram
            histogram = np.zeros(self.codebook_size)
            for word in visual_words:
                histogram[word] += 1
                
        elif self.codebook_type == "gmm":
            # Soft assignment: use posterior probabilities
            descriptors_float64 = descriptors.astype(np.float64)
            posteriors = self.codebook_algo.predict_proba(descriptors_float64)
            histogram = posteriors.sum(axis=0)
        
        # Normalize histogram
        if histogram.sum() > 0:
            histogram = histogram / np.linalg.norm(histogram)
        
        return histogram
    
    def compute_fisher_vector(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Compute Fisher Vector for a set of descriptors (only for GMM)
        
        Args:
            descriptors: Array of descriptors [num_descriptors, descriptor_dim]
            
        Returns:
            Fisher vector
        """
        if self.codebook_type != "gmm":
            raise ValueError("Fisher vectors require GMM codebook")
        
        if not self.is_fitted:
            raise ValueError("Codebook must be fitted before computing Fisher vectors")
        
        if descriptors is None or len(descriptors) == 0:
            # Return zero vector
            d = self.codebook.shape[1]
            return np.zeros(2 * self.codebook_size * d)
        
        descriptors = descriptors.astype(np.float64)
        
        # Get GMM parameters from fitted codebook
        gmm = self.codebook_algo
        Q = gmm.predict_proba(descriptors)  # Responsibilities [N, K]
        means = gmm.means_
        covs = gmm.covariances_
        
        # First order (mean gradients)
        G_mu = (Q[:, :, None] * (descriptors[:, None, :] - means[None, :, :]) / np.sqrt(covs[None, :, :])).sum(axis=0)
        
        # Second order (variance gradients)
        G_sigma = (Q[:, :, None] * ((descriptors[:, None, :] - means[None, :, :]) ** 2 / covs[None, :, :] - 1)).sum(axis=0)
        
        # Concatenate and normalize
        fv = np.hstack([G_mu.flatten(), G_sigma.flatten()])
        fv = fv / np.linalg.norm(fv)
        
        return fv


def visualize_bow_histogram(histogram, image_index, output_folder="./bovw_visualizations"):
    """
    Visualizes the Bag of Visual Words histogram for a specific image and saves the plot.
    
    Args:
        histogram (np.array): BoVW histogram.
        image_index (int): Index of the image for reference.
        output_folder (str): Folder where the plot will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Create the plot
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(histogram)), histogram, width=1.0, edgecolor='black', linewidth=0.5)
    plt.title(f"BoVW Histogram for Image {image_index}")
    plt.xlabel("Visual Word Index")
    plt.ylabel("Normalized Frequency")
    plt.grid(True, alpha=0.3)
    
    # Save the plot to the output folder
    plot_path = os.path.join(output_folder, f"bovw_histogram_image_{image_index}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Histogram plot saved to: {plot_path}")


def compare_histograms(histograms: List[np.ndarray], labels: List[str], 
                       save_path: str = "histogram_comparison.png"):
    """
    Compare multiple BoVW histograms side by side
    
    Args:
        histograms: List of histogram arrays
        labels: List of labels for each histogram
        save_path: Path to save the comparison plot
    """
    fig, axes = plt.subplots(len(histograms), 1, figsize=(12, 3*len(histograms)))
    
    if len(histograms) == 1:
        axes = [axes]
    
    for i, (hist, label) in enumerate(zip(histograms, labels)):
        axes[i].bar(range(len(hist)), hist, width=1.0, edgecolor='black', linewidth=0.5)
        axes[i].set_title(f"BoVW Histogram - {label}")
        axes[i].set_xlabel("Visual Word Index")
        axes[i].set_ylabel("Normalized Frequency")
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to: {save_path}")
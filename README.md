Demo code for ICCV19 Paper: SANet: Scene Agnostic Network for Camera Localization

### Requirements
* `CUDA 9.0`
* `OpenCV 3.2`
* `pybind11 2.4.3`
* `Python 3.6`
* `Pytorch 0.4.1` (The cuda module requires 0.4.1 to compile)
* `jupyter lab` or `jupyter notebook`
* `ipyvolume 0.6.0` (Visualizing 3D point cloud in jupyter)

 ### Preparation

 1. Download 7Scenes dataset from https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/
 2. extract all sequences zip files.
 3. Re-organize the sequences using:
     ```bash
    python seq_data/seven_scenes/scenes2seq.py <7scene_seq_dir>
    
    e.g. suppose 7scenes sequence `heads` in /home/xxx/Dcouemtns/7scenes/heads, then,
    python seq_data/seven_scenes/scenes2seq.py /home/xxx/Dcouemtns/7scenes/heads
    ```
    The above python script will generate two binary files: `train_frames.bin` and `train_frames.bin` inside of sequence directory, each file stores information of train or test frames (e.g. extrinsic and intrinsic matrix), and can be loaded wih `pickle` lib.
 4. Download pre-trained pytorch model from [Google Drive](https://drive.google.com/file/d/11vBhaFJDR5pCZKBjHr5AdQxYxLO9klBd/view?usp=sharing), unzip files to `data` folder.
    It has two pre-trained model:
    * `seven_scene_model.pth` sa-net model trained with _sun3d_ and use _7Scenes_ sequences for evaluation.
    * `netvlad_vgg16.tar` netvlad model used for query retrieval.
 5. Compile python interface for `vislearn/LessMore`:
    ```bash
    cd libs/lm_pnp
    mkdir build
    cd build
    cmake ..
    make all
    ```
    _Note_: you may need to modify the variable `PYTHON_EXECUTABLE` in line `59` of `CMakeLists.txt` with your own python interpreter. 
  6. Compile PointNet++ module (requires `pytorch 0.4.1`)
      ```bash
      cd relocal/pointnet2
      mkdir build
      cd build
      cmake ..
      make
      ```

### Example (7Scenes)
Please refer to notebook `example_7scenes.ipynb`, it requires `ipyvolume` lib for visualizing point clouds. 
  
### Todo:
Add `cambridge` dataset examples. 

### References:
1. E. Brachmann, C. Rother, ”Learning Less is More - 6D Camera Localization via 3D Surface Regression”, CVPR 2018
2.  Charles R. Qi, Li (Eric) Yi, Hao Su, Leonidas J. Guibas, "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space"

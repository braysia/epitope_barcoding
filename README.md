# epitope_barcoding

Code and sample data generated for "A multiplexed epitope barcoding strategy that enables dynamic cellular phenotypic screens". 

### Sequence analysis for linking NNNs and epitopes
```
python src/seq_analysis.py
```
This produces `nandbar.npz` which contains a list of a pair of NNN sequences and the associated epitope combinations.


### Aligning and unmixing
```
python tests/tests_align.py
python tests/tests_unmixing.py
```

### Live-cell analysis
`data` also contains sample setting files of [CellTK](https://github.com/braysia/CellTK) for tracking live single-cell trajectories.
```
data/livecell_celltk0.yml
data/livecell_celltk1.yml
```
For U-Net based segmentation, it requires pre-trained weights in HDF5 file. The hdf5 file is produced by training dataset with [CellUNet](https://github.com/braysia/cellunet).







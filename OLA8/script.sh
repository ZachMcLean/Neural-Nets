#!/bin/bash
#srun singularity exec --nv -H ${SCRATCH} /scratch/user/u.jp60244/sif/csci-2025-Spring.sif python resnet50.py augmented pretrained

#srun singularity exec --nv -H ${SCRATCH} /scratch/user/u.jp60244/sif/csci-2025-Spring.sif python resnet50.py nonaugmented pretrained

srun singularity exec --nv -H ${SCRATCH} /scratch/user/u.jp60244/sif/csci-2025-Spring.sif python resnet50.py augmented nonpretrained
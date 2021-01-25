# SLT-project

The problem of Sign Language Translation has been dealt with as two separate independent sub-problems for a long time: Sign recognition (from video to “glosses”) and Glosses Translation (from recognized glosses to understandable text). However, it has been shown that this baseline leads to a bottleneck due to the loss of information from signs to glosses. The goal of our project was to use a more recent approach avoiding this bottleneck. To enhance this approach, we tried to combine it with a pose estimation algorithm and tried different fusion strategies. We experienced a slight improvement in the performance from BLEU-4 = 20.56 without features to BLEU-4 = 21.28 with the pose estimation (on à compressed version of RWTH-PHOENIX-Weather 2014 \footnote{The dataset used can be download at \url{https://github.com/Rhythmblue/Sign-Language-Datasets}}). We also provided quantitative and quantitative analysis of the defaults of the approach and suggested future enhancements that could be investigated.



